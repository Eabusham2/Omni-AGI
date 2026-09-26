# Neural substrate persistence

Stable v1 persists the growable neural substrate outside the monolithic
`plasticity.safetensors` checkpoint. The in-memory substrate is still one
authoritative neural state; sharding is only its bounded persistence format.

## Layout

```text
engine/substrate/
  manifest.json
  generations/<content-sha256>/manifest.json
  forward-index/generations/<content-sha256>.json
  blobs/<sha256>.json
  blobs/<sha256>.safetensors
```

`manifest.json` is an atomically written convenience pointer. The substrate
generation embedded in `engine/brain.json` is the complete-brain commit record:
load, copy, and export follow that embedded generation even if an interrupted
save left a newer orphan root pointer. A generation manifest is immutable and
declares every bounded record and tensor shard by relative path, kind, hash
bucket, part, record count, byte count, and SHA-256. JSON shards hold neuron,
assembly, and synapse structure. Safe-tensor shards hold neuron/assembly
hypervectors and the higher-precision learning state for sparse synapses.
Effective sparse weights reload only when they are exactly `-1`, `0`, or `+1`.

Large generations also publish an atomic, internally checksummed forward index
bound to the exact generation content hash, generation-manifest checksum, shard
checksums, counts, and assembly set. It contains no source text: only exact
nonzero endpoint structures, canonical two-bit ternary levels, per-shard ID
ranges, and adjacency locators for connected assembly records. A valid index
lets cold startup avoid opening every synapse JSON/safe-tensor pair. Any record
page-in still verifies its content-addressed shard before use, and
`scrub_persisted_shards()` performs an uncapped full verification plus
source-shard/index parity proof for explicit or background maintenance. Legacy
generations can build the same index once with
`scripts/build-substrate-forward-index.py`.

Shards use deterministic hash buckets and content-addressed names. An unchanged
save reuses every shard. A local growth update normally replaces only the
affected bucket and the small generation manifests; it does not rebuild a
single all-neuron or all-synapse tensor. A bucket can split into further parts,
so this format has no model-defined neuron, assembly, synapse, or shard-count
ceiling.

The writer checks the host RAM/disk reserve before creating each new blob and
before promoting a generation. A pause leaves the prior `brain.json` generation
loadable. Completed but unreferenced content-addressed blobs are harmless and
can be reclaimed by a future offline garbage collector.

Snapshots, immutable origins, forks, neural-evolution rollback points, and
`.omni` archives copy a synthesized pointer for the committed generation, its
generation manifest, and only the blobs declared by that generation. Every path
and checksum is verified before the destination pointer or `brain.json` is
promoted. `.omni` export and import stream these shards through ZIP/ZIP64 and
have no product-defined archive byte or entry-count ceiling; the disk reserve
and platform filesystem limits still apply.

## Live activation precision

Sparse recurrent activation uses the effective ternary value directly:

```text
edge contribution = source activation * {-1, +1}
settled target = seed + 0.52 * mean(incoming edge contributions)
```

The exact ternary edge is the learned weight; short-lived eligibility and
timing state can influence when it changes but are not a second stored weight.
Negative edges and negative recurrent signals enter the
same settled state as competing inhibition. Fan-in normalization makes the
recurrent operator contractive, so it settles without a fixed hop limit.
Operational traces report ternary coverage, inhibitory signals, suppressed
assemblies, convergence delta, and settling rounds.

## Stable-v1 limitations

- Neurons, assemblies, vectors, the exact nonzero recurrent graph, and touched
  synapse records are resident while a brain runs. Untouched synapse records
  remain shard-backed and are paged exactly on demand.
- Content-addressed generations are retained for rollback. Automatic
  mark-and-sweep garbage collection is not part of v1.
- Hash-bucket updates bound rewrite scope; an especially hot bucket may still
  rewrite one or more bounded parts.
- Early monolithic-substrate and public beta checkpoints are intentionally
  incompatible with the current whole-brain importer. They are not rewritten
  or relabeled as native sharded OmniCortex state.
