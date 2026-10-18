# `.omni` portable brain format

## Version 2 status

An `.omni` file is a non-executable ZIP container for OmniCortex state. New
exports use format version 2, with complete saved whole-brain catalog custody:

```json
{
  "format": "omni-brain",
  "formatVersion": 2,
  "requiredFeatures": ["whole-brain-catalog-closure-v1"],
  "architecture": "OmniCortex",
  "architectureSchemaVersion": 1
}
```

The schema-version value above is illustrative of the current build constant;
import requires an exact match with that build. Unknown required features are
rejected before any identity adoption. Older apps are not an export
compatibility constraint. The importer never loads pickle objects and never
executes repository code or setup scripts from a bundle.

Read-only format-1 import is limited to otherwise-valid, currently supported
native OmniCortex state without whole-brain catalog pointers or a catalog
payload namespace. Catalog-bearing or dangling format-1 archives are rejected
with a format-2-required error. Typed API-teacher media custody also requires
format2 and its conditional required-feature declaration below. This is not a foreign-model or legacy-hybrid
compatibility mode, and existing external files are never rewritten or erased.

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

Every version-2 bundle contains both a selected payload and an immutable-origin payload:

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

Both selected and immutable-origin states also require their exact saved
whole-brain catalog closure, including reachable retained lineage:

```text
catalog/current/whole-brain-catalog/blobs/<sha256>.json
catalog/current/whole-brain-catalog/index/<sha256>.sqlite3
catalog/current/whole-brain-catalog/lineage/blobs/<sha256>.json
catalog/current/whole-brain-catalog/lineage/sources/<entry-sha256>/whole-brain-catalog/...
catalog/origin/whole-brain-catalog/...
```

Only the exact chain and immutable location index named by saved metadata are
included; local orphan/history files are not globbed into a bundle. A direct
catalog pointer binds the source identity, committed core/substrate generations,
attention-overlay digest and both connected-region descriptor digests. Every
catalog record and retained lineage entry keeps its original source identity.
Missing or empty source registries are rejected; export never loads a model,
invents an empty map, or rewrites the immutable origin to fill this gap.

A copied checkpoint may have no current destination registry while preserving
the real source registry as lineage. In that case its archive includes
`catalog/current/projection.json` (or the corresponding origin scope when
applicable). This checksummed, archive-only `preserved-current-state`
projection binds the actual saved owner/state to one exact retained source
entry. It requires an exact match of all non-owner state-binding fields;
distinct matching registries are ambiguous and rejected. Identical source
pointers may be deduplicated for selection, but all lineage provenance stays.
Import re-derives the projection rather than trusting its claim. It is not
persisted as a live destination registry or activation record. Subsequent
copy/re-export derives it again without a model Load or source mutation.

When saved current, origin, recovery or connected-parent state contains MCP
task receipts, the manifest additionally requires `mcp-task-custody-v1`:

```text
mcp-tasks/current/operational-mcp-tasks/<task-ref>.json
mcp-tasks/current/operational-mcp-tasks/history/<sha256>.json
mcp-tasks/current/operational-mcp-tasks/payloads/<sha256>.json
mcp-tasks/current/operational-mcp-tasks/lineage/<closure-sha256>/...
mcp-tasks/origin/operational-mcp-tasks/...
```

Recovery points retain the same exact paths beneath their saved `engine/`;
connected-parent payloads remain opaque ZIP blobs whose authenticated logical
paths are checked as complete task closures. Current folders are first
captured coherently under a short canonical task-store lease, then immutable
staged files stream without a model-export-long lease. Import validates exact
record/history/content addresses and final payload receipts before reserving
an identity. Unknown files, missing payloads, injected live-authority
fields or missing feature declarations fail closed. Original task IDs, source
owners, observed receiver IDs and raw persisted content remain unchanged.
The records are historical, not live server sessions, keys, request leases,
permissions or replay grants. Copies/imports cannot regain task control merely
from matching an old ID. See [MCP task custody](v2/mcp-task-portability.md).
Known interrupted-publication `.json.<UUID>.next` fragments are retained as
exact inert bytes without parsing or promoting them into task state. Arbitrary
unknown partial-file names still fail validation.

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

The checksum-bound `learning-workspace/` inventory also carries host-owned
current-task attachment registries and their content-addressed raw blobs. An
attachment record binds brain, turn, immutable handle, MIME, byte length,
content hash, private source provenance, and state. External import/fork rekeys
the current owner without rewriting the original source bytes; recovery selects
the saved registry; full merge imports parent attachments only as inert
attributed history. Unsent handles explicitly removed before binding are not
portable. Attachment staging alone is not a dataset or neural-learning receipt.

When the learning inventory retains typed API-teacher media requests, received
response envelopes, lessons or assets, the manifest additionally requires
`api-teacher-typed-media-v1`. This is a supported conditional feature, not a
mandatory token for teacher-free archives. Exact logical custody includes:

```text
api-teachers/workspace.sqlite3
api-teachers/archives/<sha256>.sqlite3
api-teachers/envelopes/<sha256>.json
api-teachers/assets/<sha256>.<verified-media-extension>
api-teachers/lesson-<sha256>.jsonl
```

These logical files are carried by the existing learning-workspace inventory
and checksummed blobs, including historical recovery inventories and opaque
connected-parent `portable-learning-workspace` inventories. The importer checks
the complete current/sealed-history teacher reference closure before reserving
an identity; missing declarations, files, byte counts, signatures or hashes
refuse adoption. Current and origin export modes preserve this unsanitized
operational history. A same-ID external import still makes copied teacher jobs
dormant; it does not turn a saved job into provider authority merely because the
ID was unused. Explicit same-owner recovery preserves its genuine selected
paused prefix instead. Credentials, live leases and grants are not copied.
See [typed teacher media](v2/api-teacher-media.md) for exact states and evidence.

Current, private-archive, and referenced-local bundles with declarative plugin
history additionally carry:

```text
plugins/manifest.json
plugins/packages/<content-sha256>.json
```

The plugin manifest binds each immutable package, its public documentation,
reviewed selection, ordinary semantic-learning receipt, and install provenance.
Origin-only bundles omit later mutable plugin selections. Installation
credential-store records, live sessions, approval tokens, and catalog database
paths are never plugin payloads. User-authored documentation is preserved
exactly and may itself contain sensitive text, so it still requires review.

## Manifest contract

The manifest records and validates:

- exact `format`, `formatVersion`, `architecture`, and `architectureSchemaVersion`;
- format-2 `requiredFeatures`, including complete catalog-closure support;
- conditional typed API-teacher media and MCP task-custody feature requirements;
- export timestamp, brain ID, display name, and lineage;
- one declared export mode;
- materialized-engine status, memory recipe, raw-episode status, and `ternary-effective` quantization;
- the current and immutable-origin packed-ternary manifest hashes, tensor
  counts, and—only for referenced-local mode—per-file object references;
- descriptors and checksums for the current and immutable-origin substrate
  pointers, generation manifests, and every referenced bounded shard;
- complete current/origin catalog chains, immutable indexes, retained lineage
  closures and any exact re-derived archive-only copied-state projection;
- `savedInstance: {version: 1, content: "unsanitized"}`, zero replacements in
  the retained `secretRedaction` envelope, and included conversation state;
- declared recovery points and any exact already-missing historical references;
- when present, one plugin-portability declaration with exact selection/package
  counts and the SHA-256 of `plugins/manifest.json`;
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
- rejects unknown required features before adoption; requires the exact catalog
  current/origin/lineage namespace with no missing or orphan payloads, validates
  every catalog checksum and index-to-record location, and re-derives copied
  current-state projections from the authenticated metadata;
- validates exact typed API-teacher closure in ordinary, historical and opaque
  connected-parent inventories before identity adoption, with selected-family,
  physical and heap parsing admission;
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
- validates plugin package canonical hashes, declarations, selection/provenance
  identity and semantic receipt hashes; rejects undeclared plugin files,
  unreferenced packages, and credential-like immutable delegate constants;
- validates replay SQLite integrity, every tensor-row payload checksum, and
  the declared committed count/high-water/content digest before publishing an
  imported identity; pending rows are validated too;
- resolves referenced tensors only from the destination repository's local object store and then validates the resolved safe tensors.

If an imported brain ID already exists, the importer assigns a new ID and
advances lineage instead of overwriting the existing brain. Extraction and
validation occur in temporary storage; any failed check removes the candidate
before it becomes a visible brain.

An external import installs verified missing plugin dependencies inactive and
stores selections as dormant history. It creates no per-instance plugin access
and imports no grant. Trusted local duplicate/fork is a different authority
boundary: it may preserve the exact current reviewed selection while giving the
new identity independent host-policy revisions. Full merge preserves each
parent's plugin selection as a distinct dormant provenance record.

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

Catalog JSON is read through an admitted, exact-sized file descriptor before
decoding. Physical free memory, available V8 heap and the host's selected app
RAM ceiling must admit its conservative scratch requirement. Source size,
device/inode, timestamps, canonical path and checksum are rechecked around the
read. JSON shards are traversed one at a time; SQLite uses a bounded page cache
and file-backed temporary work, not a full index-to-array load. These are
cooperative allocation checks, not a claimed OS-enforced whole-process RSS
limit. Resource pressure pauses/refuses the operation; there is no fixed total
brain, archive, record or lineage-count product cap.

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
