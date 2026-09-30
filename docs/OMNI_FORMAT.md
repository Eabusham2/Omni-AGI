# `.omni` portable brain format

## Version 1 status

An `.omni` file is a non-executable ZIP container for OmniCortex state. The current contract remains format version 1:

```json
{
  "format": "omni-brain",
  "formatVersion": 1,
  "architecture": "OmniCortex",
  "architectureSchemaVersion": 1
}
```

The schema-version value above is illustrative of the current build constant; import requires an exact match with that build. The importer never loads pickle objects and never executes repository code or setup scripts from a bundle.

This format carries a brain's saved neural identity and immutable origin.
The local source correction preserves saved content **unsanitized**, including
chat ledgers, pending chat-learning jobs, committed ingestion cursors and cold
working-memory pages. This is saved-state preservation, not an in-flight
process clone or automatic relocation/resumption of external datasets. The
correction is not part of the already-published v1.1.0 artifacts.
An unfinished/unready instance or missing materialized origin is rejected
before export rather than producing an archive that native import cannot load.
Build recipes and modality-only weights use smaller, separate
contracts documented in [CATALOG_FORMATS.md](CATALOG_FORMATS.md).

## Required entries

Every version-1 bundle contains both a selected payload and an immutable-origin payload:

```text
manifest.json
model-card.md
checksums.sha256
state/brain.json
state/engine.json
tensors/core.safetensors
tensors/plastic.safetensors
origin/state/brain.json
origin/state/engine.json
origin/tensors/core.safetensors
origin/tensors/plastic.safetensors
```

When `engineMaterialized` is true, both exact inference payloads are also
required:

```text
packed/current/manifest.json
packed/current/manifest.sha256
packed/current/ternary-<index>-<digest>.bin
packed/origin/manifest.json
packed/origin/manifest.sha256
packed/origin/ternary-<index>-<digest>.bin
```

Each materialized payload also carries the exact committed growable-substrate
generation selected by its worker metadata:

```text
substrate/current/manifest.json
substrate/current/generations/<content-sha256>/manifest.json
substrate/current/blobs/<sha256>.json
substrate/current/blobs/<sha256>.safetensors
substrate/origin/manifest.json
substrate/origin/generations/<content-sha256>/manifest.json
substrate/origin/blobs/<sha256>.json
substrate/origin/blobs/<sha256>.safetensors
```

Only blobs referenced by the selected generation are included. The portable
root pointer is synthesized from the generation recorded in
`state/engine.json` or `origin/state/engine.json`; an uncommitted orphan root
pointer left by an interrupted save cannot change the exported identity.

`state/brain.json` is inspectable Electron state: configuration, lineage,
messages, traces, journal, derived concept/synapse summaries, and saved source
metadata without redaction. `state/engine.json` is Python-worker metadata. Core
safe tensors contain slow neural and modality parameters. Plastic safe tensors
contain non-substrate SNN state, replay tensors, liquid activity, and other
mutable recurrent state.

The substrate tree is the authoritative associative-memory payload. Its JSON
shards contain neuron, distributed-assembly, and signed ternary synapse
records. V3 safe-tensor shards contain authoritative packed neuron/assembly
vector rows; higher-precision transient activity/eligibility is not a learned
floating weight shadow. The sharded persistence contract is
documented in [SUBSTRATE_PERSISTENCE.md](SUBSTRATE_PERSISTENCE.md).

Each `packed/**` directory is a complete `omni-packed-ternary` inference
package. Its manifest enumerates every eligible neural projection plus the
dynamically grown substrate synapses, records each shape and per-tensor scale,
and requires complete coverage. The binary shards use deterministic two-bit
codes, least-significant pair first: `00 = -1`, `01 = 0`, `10 = +1`; `11` is
reserved and rejected. Unused pairs must use canonical zero padding (`01`).
`manifest.sha256` authenticates the exact canonical manifest bytes.

Every mode preserves source blobs actually referenced by its saved state and
recovery history; it does not resurrect deleted or unretained training files.
Referenced-local bundles keep required paths but replace their four
safe-tensor entries with valid placeholder safe tensors and every
packed-ternary entry with a local-reference marker. The manifest points to the
real content-addressed objects held by the originating local repository.

Saved-instance bundles additionally carry conversation/activity ledgers,
annotated artifact ledgers/files, referenced working-page databases, committed
joint-generation manifests/index snapshots and declared recovery history.
Every payload is checksum-bound. A missing historical cold-page/joint payload
is explicitly listed, not recreated or described as a restorable checkpoint.

## Manifest contract

The manifest records and validates:

- exact `format`, `formatVersion`, `architecture`, and `architectureSchemaVersion`;
- export timestamp, brain ID, display name, and lineage;
- one declared export mode;
- materialized-engine status, memory recipe, raw-episode status, and `ternary-effective` quantization;
- the current and immutable-origin packed-ternary manifest hashes, tensor
  counts, and—only for referenced-local mode—per-file object references;
- descriptors and checksums for the current and immutable-origin substrate
  pointers, generation manifests, and every referenced bounded shard;
- `savedInstance: {version: 1, content: "unsanitized"}`, zero replacements in
  the retained `secretRedaction` envelope, and included conversation state;
- declared recovery points and any exact already-missing historical references;
- an application-license declaration and normalized per-source provenance/license ledger;
- SHA-256 and exact byte length for every payload entry except `manifest.json` and `checksums.sha256`;
- for referenced-local mode, the SHA-256 object IDs for current and origin
  core/plasticity tensors and every current/origin packed-ternary file.

Every source record must make its redistribution status visible. A source without declared licensing is labeled `Undeclared; verify before redistribution`; absence of a declaration is not converted into permission.

## Integrity and import validation

Before materializing a brain, the importer:

- parses ZIP and ZIP64 central directories without buffering the complete
  archive or its expanded payloads;
- applies no product-defined compressed-byte, expanded-byte, per-entry, or
  entry-count ceiling; filesystem addressability and safe-integer limits remain
  real platform boundaries;
- checks projected extraction against the configured free-disk reserve and
  pauses/fails before consuming that reserve;
- rejects absolute paths, drive-prefixed paths, backslashes, NULs, and `.`/`..` components;
- rejects duplicate paths, overlapping payloads, inconsistent local/central
  headers, encrypted entries, symbolic links, multi-disk archives, and
  unsupported ZIP compression;
- streams stored or deflated entries to an isolated temporary directory while
  validating CRC and declared compressed/expanded lengths;
- rejects common executable/script/library extensions;
- requires every payload entry to have a descriptor and checksum;
- compares both the manifest SHA-256 and exact byte count with the streamed
  extracted file;
- requires the exact supported architecture, schema, export mode, redaction policy, and license-ledger shape;
- requires both current state and immutable origin to prove native ground-up
  OmniCortex (`origin_kind: ground-up`, no foundation model/adapter, no retired
  Starter manifest, and no random-only Blank or legacy-hybrid marker);
- parses every required JSON document;
- validates all four final safe-tensor headers and data offsets without deserializing code, after resolving local references when applicable;
- requires both complete packed-ternary packages for every materialized stable
  brain, rejects undeclared or extra pack files, and validates manifest,
  shard, packed-payload, decoded-tensor, reserved-code, canonical-padding, and
  eligible-projection coverage checksums before materializing the brain;
- follows each engine metadata commit to one exact substrate generation,
  validates its content hash, kind/bucket/part/count descriptors, every
  content-addressed blob name and checksum, and the effective ternary value of
  every sparse synapse;
- verifies content-addressed object names against their bytes;
- validates replay SQLite integrity, every tensor-row payload checksum, and
  the declared committed count/high-water/content digest before publishing an
  imported identity; pending rows are validated too;
- resolves referenced tensors only from the destination repository's local object store and then validates the resolved safe tensors.

If an imported brain ID already exists, the importer assigns a new ID and
advances lineage instead of overwriting the existing brain. Extraction and
validation occur in temporary storage; any failed check removes the candidate
before it becomes a visible brain.

## Export modes

| API mode | Manifest mode | Selected payload | Portability |
| --- | --- | --- | --- |
| `current` | `current-portable` | Current saved neural/chat/working/continuation state and recovery history, unsanitized | Self-contained saved state; external datasets not relocated |
| `origin` | `origin-portable` | Selected immutable starting checkpoint plus its saved state and declared recovery history | Self-contained saved state |
| `private-archive` | `private-archive` | Current saved state and referenced retained source content, unsanitized | Self-contained saved state and sensitive |
| `referenced` | `referenced-local` | Unsanitized current saved state with local tensor references | Same repository only |

Every mode includes an `origin/**` payload and both `packed/current/**` and
`packed/origin/**`, plus `substrate/current/**` and `substrate/origin/**`, when
the neural engine is materialized. In a
referenced-local bundle the exporter emits valid safe-tensor placeholders and
packed-file reference markers until import resolves every declared hash.
Export first stores the real tensor and packed bytes in the repository
`.blobs` store. A different installation without those exact objects rejects
the import, so referenced-local files must not be advertised as portable or
shareable checkpoints.

## Streaming and resource boundary

The exporter writes each in-memory metadata entry or regular-file payload
directly to a temporary archive, calculates CRCs while streaming, and emits
ZIP64 fields only when classic ZIP counts, offsets, or lengths overflow. It
atomically replaces the requested destination after the complete central
directory is durable. It never accumulates a complete tensor, packed shard,
substrate generation, or `.omni` file in one buffer.
Current and origin replay databases are first staged through a SQLite snapshot
that includes committed WAL pages; the same staged file is validated, hashed,
and streamed into ZIP. An open writer therefore cannot leave a committed or
pending replay row behind in an omitted WAL sidecar.

The importer reads central metadata with random access, rejects unsafe
structures before extraction, and streams each payload through CRC/length
validation into a temporary directory. “No fixed archive cap” does not mean
infinite storage: host filesystem limits, available address space, configured
free-disk reserve, user cancellation, and corrupt or unsupported ZIP structures
remain explicit stopping conditions.

## Saved content and privacy

Every mode preserves saved text, source paths, settings, annotations, retained
blobs and neural state without credential-pattern redaction. Private paths or
credential-shaped text are not silently removed or refused merely for being
private. JSON/text, hashes, native origins, safetensors, archive paths and
reference integrity remain validated. A passing export is **not** a
secret-free certificate: the desktop warns before all modes, and sharing must
be limited to trusted recipients.

The bundle excludes the app's external browser partition and OS credential
vault. Saved brain content may itself contain secrets; that is preserved as
requested. Import begins dormant rather than automatically performing external
actions. A collision rekeys live ownership while preserving the immutable
origin and rehashing imported recovery ownership.

Pending learning receipts and dataset cursors are preserved, not automatically
executed. Dataset continuation still checks its original absolute path/stat,
content hash, name/kind/policy/epoch and parser bindings. Moving the source
elsewhere does not silently reset or remap its cursor. Durable neural-ledger
rows beyond the authoritative checkpoint are archived as present; the worker's
existing authorized-load recovery discards the uncommitted suffix. No old
missing snapshot payload is fabricated. Export does not modify the source
mind's learned state or queued records.

## Provenance and licensing

The manifest's license ledger preserves the Omni application license declaration plus source name, optional provenance URL, license label, and optional license URL. Saved source content hashes, timestamps, policies and provenance are preserved without a privacy projection. Import requires a structurally valid ledger, and the desktop can expose those declarations before a pack is installed. License metadata is attribution and warning data; it does not itself verify ownership or grant rights.
