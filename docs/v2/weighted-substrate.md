# Weighted native connections: implementation and evidence

Updated 2026-10-02. This note covers the native projection changes in this lane. It does not mark the complete v2 plan, sparse-router extension, large-model performance, or learned capability as complete.

## Representation and computation

`PackedAdaptiveBitLinear`, `PackedAdaptiveTernaryEmbedding`, and all native 1D/2D/3D standard/transposed convolutions now have:

- Their existing authoritative two-bit ternary sign stream: `00=-1`, `01=0`, `10=+1`; `11` is invalid and unused padding remains zero-sign.
- A `uint8` `_connection_state` matrix. Its low four bits encode integer strength minus one (strength 1–16). Its high four bits encode resistance (0–15). Biased projections also have `_bias_connection_state`.
- Independent repeated contacts in `parallel_connections.pages`, with canonical public IDs `w_r<row>_c<column>_n<ordinal>` or `b_...`. Contacts share 1024-slot packed pages rather than individual Python modules. Each slot has a two-bit sign and one strength/resistance byte; opposing contacts remain independent.
- Existing row-resistance state. Successful discrete changes increase connection resistance; row resistance remains an additional local stability term.

The effective contribution is `sign × strength`, plus any repeated contacts at that endpoint, then the existing projection scale. The sign is ternary; the resulting summed integer coefficient can exceed ±1. Strength codes are learned directly and are not a floating-point master weight.

Forward and input-gradient paths consume the same state for linear, embedding, convolution and transposed convolution. The loss backward path changes sign/strength bytes using finite transient derivatives and seeded stochastic one-step changes. Saturated strength may create another independently learned contact after resource admission. Contacts born during an update are not visited again during that update. Canonical numeric contact ordering preserves seeded behavior after save/load.

The native projection objects expose no floating `nn.Parameter` weight array or Adam moments. Activations, transient derivatives and existing scalar controls retain their appropriate arithmetic types.

Linear/convolution forward and input gradients now call the direct packed CPU/Metal kernels supplied by the root lane. They decode signs and strengths inside the kernel, without constructing expanded weight rows. Unsupported accelerator calls transfer admitted, bounded raw-packed CPU tiles. Repeated contacts add bounded activation-vector contributions; embedding processes only touched rows. The obsolete decoded-weight matrix fallback helpers were removed.

## APIs and persistence

- `connection_strengths(state)`, `connection_resistance(state)`, `encode_connection_state(strength, resistance)`, and `weighted_ternary_levels(levels, state)` define the compact representation.
- Projection APIs: `set_connection_strengths_(values, bias=False)`, `connection_strengths(bias=False)`, `add_parallel_connection_(row, column, sign=0, strength=1, bias=False)`, and `iter_parallel_connection_records(...)` / `parallel_connection_records(...)`.
- `install_parallel_connection_records_(...)` installs a bounded stream batch without deleting earlier batches. `copy_parallel_connections_from_(source)` copies exact pages into a fresh nonshrinking projection, preserving IDs, signs, resistance and strengths during region growth.
- `authoritative_packed_tensors()` includes the owner's base sign/connection buffers; traversal also includes the pool identity counter and each page's signs, strengths, endpoints and live-slot bytes exactly once. Structural bytes affect identity/checksums and rollback; they are excluded from the weight-magnitude delta norm.
- `capture_packed_topology_boundary()` / `restore_topology_boundary()` integrate contact birth with `PackedMutationSnapshot`, bounded disk rollback, and the parameter-delta journal.
- `NativeCorePager.retire(module)` unregisters a removed page and unlinks only that pager's temporary backing files. Existing tensor views remain valid; Windows may retain a mapped backing file until its handle closes, as with existing pager cleanup.
- `packed_forward_status()` reports sign bytes, connection state bytes, repeated-contact learned bytes, the connection encoding, and the separate registry admission estimate.

Pre-strength native checkpoints receive zero connection bytes, meaning unit strengths and zero connection resistance, with unchanged initial computation. They receive an empty contact pool and zero identity counter. Partial connection state, incomplete pages, or pages without the base connection state/counter are rejected. This is native format migration, not support for a foreign model backend.

Pool state uses `_next_page: uint8[8]`. Page keys encode role, a 32-row lookup bucket, and a global page ID. Each page stores `_packed_sign: uint8[1,256]`, `_connection_state: uint8[1,1024]`, and `_record_meta: uint8[1024,10]` (row offset, presence flag, uint64 little-endian column). The ordinal is `pageID × 1024 + slot`. Removed committed slots become tombstones and are not reused. Transaction rollback can discard uncommitted tail slots/pages. A separate nonpersistent `_sort_order: uint8[1024,2]` is a reconstructible page index included in pager residency, but excluded from checkpoints and neural checksums.

Safe-tensor checkpoints include all registered contact state. Strict PyTorch and bounded state loads prepare required child owners; paged children are usable when loading completes. Malformed restored endpoints, identities, ranges or codes are rejected before contact replacement begins.

## Packed inference format 2

The sign shard remains the canonical packed ternary stream. Each tensor also carries either explicit unit-strength metadata or an authenticated connection shard:

```json
{
  "connectionState": {
    "formatVersion": 1,
    "encoding": "uint4-strength-minus-one-high4-resistance",
    "shape": [3, 7],
    "shard": "connections-00000-<digest>.bin",
    "byteLength": 21,
    "sha256": "<full digest>"
  }
}
```

Ordinary matrix/conv state shape is `[shape[0], product(shape[1:])]`; a bias vector uses `[1, shape[0]]`. Transposed convolutions explicitly add `packedRowLayout: {kind: "transpose-convolution", groups: G}`. Their sign-stream and connection order is the native matrix `[shape[1] × G × product(shape[2:]), shape[0] / G]`; the declared tensor shape still records logical convolution geometry. The checksum authenticates the native ordered sign stream under that declared shape. Consumers must use the authenticated layout, not treat it as an ordinary dense convolution array.

Repeated contacts use a companion file of strictly ordered 26-byte records: little-endian `uint64 row`, `uint64 column`, `uint64 ordinal`, `uint8 packedSign`, `uint8 connectionState`. Export streams a `ParallelConnectionSource` and does not collect all contacts into a Python list. Checks cover hashes, lengths, geometry, sign/padding codes, ordered unique identities, and unexpected or reused shards. Selective verification still verifies unretained connection and contact shards. Connection export detects concurrent state mutation. Retained connection state is returned from the verified stream instead of rereading unverified bytes.

`VerifiedTernaryBundle` now returns `connection_states` and `parallel_connections` alongside sign tensors and the manifest. Root integration owns `.omni` installation, desktop validation, whole-brain accounting, and lineage/export decisions.

## Mutation, distributed state, and resources

Local, loaded and replica-installed page allocations share one admission path. It reserves an estimated 8 KiB of Python/registry allocation per page, plus separately admitted native buffer/backing allocations. A page holds up to 1024 contacts, with no fixed number of pages. A refused allocation does not silently drop a connection or invent success.

Distributed updates stage bounded derivatives, apply one seeded canonical update, and share authoritative signs, strengths/resistance, and repeated contacts. Identity hashes include endpoints, ordinals and structural allocation state. Rollback restores exact bytes and removes newborn contacts/pages. Contact rollback is stored in SQLite chunks and transfers use at most 128 contact records per batch, including when one row has many contacts. Identity hashing and inference export stream records. Forward traversal merges one record cursor per selected page, rather than retaining every contact.

Raw storage and runtime cost are distinct:

| Component | Cost |
| --- | --- |
| Base sign | Two physical bits per logical sign, plus row padding |
| Base strength/resistance | One byte per base contact |
| Repeated learned state | 256 sign bytes + 1024 strength/resistance bytes per allocated page |
| Repeated endpoint/live metadata | 10240 bytes per allocated page, plus one 8-byte counter per pool |
| Repeated ordering cache | 2048 bytes per allocated page; reconstructible and excluded from checkpoints |
| Repeated inference export | 26 bytes per record, including endpoint/ordinal metadata |
| Repeated-contact host registry | One Python module per page; 8 KiB admission estimate per page |
| Cold storage backing | Actual filesystem/mapping allocation, reported separately by the pager |

The 8 KiB estimate is an admission allowance, not a measured constant. `parallelConnectionRegistryStorage` reports `packed-record-pages`; `parallelConnectionBackingAllocationIncluded` is false. Stats report allocated page capacity, so a partly filled page is not mislabeled as only its active slots' bit cost.

## Verification completed in the first correctness pass

One final affected run passed **84 tests** in 3.647 seconds:

```sh
PYTHONPATH=engine .venv/bin/python -m unittest \
  engine.tests.test_weighted_native_connections \
  engine.tests.test_packed_adaptive_linear \
  engine.tests.test_packed_adaptive_embedding \
  engine.tests.test_packed_adaptive_conv \
  engine.tests.test_ternary_packing \
  engine.tests.test_native_core_paging \
  engine.tests.test_packed_collective \
  engine.tests.test_parameter_diagnostics \
  engine.tests.test_packed_mutation_snapshot -q
```

Coverage includes exact forward gains/input gradients; all convolution dimensions and both directions; a tiny loss-backward fixture decreasing loss from 4 to 1 to 0 through learned strength bytes; unit migration; saved-state and inference companion round trips; corruption rejection; canonical seeded replay; contact birth admission; RAM/disk/journal rollback; aborted small-head transactions; and actual two-rank Gloo synchronization with exact rollback. The Gloo fixture used loopback after a hostname-resolution warning.

A separate real MPS check on this host produced output `[[1.0]]` and input gradient `[[4.0, -3.0]]` for a weighted/repeated two-input projection and reported an empty learned-parameter list. It is a small functional check, not a speed benchmark or broad MPS training qualification.

`git diff --check` passed after these changes. No saved user brain or full corpus was built or trained; the tests instantiate only tiny temporary native components.

## Pooled-storage and direct-kernel follow-through

The additional small fixtures verify more than 1024 contacts crossing a page boundary, bounded saved-state loading into cold pages, stable IDs across deletion/reload, deep-copy isolation, streamed export, exact transfer into a wider region, rewiring-sensitive checksums, and forward/backward with weight-unpacking and Torch matmul patched to fail. A two-rank fixture crosses the 128-record collective transfer boundary and verifies rollback.

The final affected command passed **96 tests in 5.290 seconds**, spanning the original nine files plus `test_pooled_parallel_connections`, `test_weighted_brain_accounting` and `test_packed_integer_kernel`. This overlaps and supersedes the earlier 84-test pass; the totals must not be added together. One subsequently added nonalphabetical-page-order checksum/reload test passed separately in 0.037 seconds. Python compilation and `git diff --check` also passed.

On this Mac, the integrated model path using real Metal returned `[[1.0]]` with gradient `[[4.0, -3.0]]`, while the contact pool remained CPU/pager-backed. This does not establish large-model throughput.

## Still required by the wider v2 plan

1. **Large-scale performance measurements.** Pooled storage and direct CPU/Metal routing are implemented. Full-size peak memory, sustained paging, million-contact update rate and multi-device transfer throughput still require measured qualification. No such benchmark is implied by the short fixtures.
2. **Other hardware backends and release packaging.** The root kernel lane owns CUDA/other acceleration and bundling the native libraries. Unsupported accelerator fallback remains direct packed CPU computation; it does not reinstate decoded weights.
3. **Weighted sparse-router integration.** That separate lane owns sparse STDP state and its evidence. This lane supplied strength-byte export and sparse zero-birth diagnostics, but its projection tests do not certify the whole router.
4. **Whole-brain and product evidence.** The root lane owns full accounting, resource estimates, desktop import/export, merge preservation, viewer integration, and broader audits. Full trained quality/retention remains outside these component checks.
