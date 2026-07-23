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

This format is for a complete identity. Build recipes and modality-only weights deliberately use smaller, separate contracts documented in [CATALOG_FORMATS.md](CATALOG_FORMATS.md).

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

`state/brain.json` is inspectable Electron state: configuration, lineage, messages, traces, journal, concept/synapse summaries, and sanitized or archived source metadata. `state/engine.json` is Python-worker metadata. Core safe tensors contain slow neural and modality parameters; plastic safe tensors contain SNN state, VSA vectors, replay tensors, liquid activity, and other mutable recurrent state.

Private archives may add `blobs/<sha256>` source objects. Referenced-local bundles keep all required paths but replace their four tensor entries with valid placeholder safe tensors; the manifest points to the real content-addressed tensor hashes held by the originating local repository.

## Manifest contract

The manifest records and validates:

- exact `format`, `formatVersion`, `architecture`, and `architectureSchemaVersion`;
- export timestamp, brain ID, display name, and lineage;
- one declared export mode;
- materialized-engine status, memory recipe, raw-episode status, and `ternary-effective` quantization;
- secret-redaction policy version and replacement count;
- an application-license declaration and normalized per-source provenance/license ledger;
- SHA-256 and exact byte length for every payload entry except `manifest.json` and `checksums.sha256`;
- for referenced-local mode, the SHA-256 object IDs for current and origin core/plasticity tensors.

Every source record must make its redistribution status visible. A source without declared licensing is labeled `Undeclared; verify before redistribution`; absence of a declaration is not converted into permission.

## Integrity and import validation

Before materializing a brain, the importer:

- limits the compressed container to 512 MiB;
- limits each expanded entry to 512 MiB and total expanded data to 1 GiB;
- limits the central directory to 4,096 entries;
- rejects absolute paths, drive-prefixed paths, backslashes, NULs, and `.`/`..` components;
- rejects duplicate paths, encrypted entries, symbolic links, and unsupported ZIP compression;
- rejects common executable/script/library extensions;
- requires every payload entry to have a descriptor and checksum;
- compares both the manifest SHA-256 and exact byte count with the extracted bytes;
- requires the exact supported architecture, schema, export mode, redaction policy, and license-ledger shape;
- parses every required JSON document;
- validates all four final safe-tensor headers and data offsets without deserializing code, after resolving local references when applicable;
- verifies content-addressed object names against their bytes;
- resolves referenced tensors only from the destination repository's local object store and then validates the resolved safe tensors.

If an imported brain ID already exists, the importer assigns a new ID and advances lineage instead of overwriting the existing brain. Any failed check aborts import before the candidate becomes a brain.

## Export modes

| API mode | Manifest mode | Selected payload | Portability |
| --- | --- | --- | --- |
| `current` | `current-portable` | Current evolved state | Self-contained |
| `origin` | `origin-portable` | Immutable starting state | Self-contained |
| `private-archive` | `private-archive` | Current state plus retained source blobs after confirmation | Self-contained and sensitive |
| `referenced` | `referenced-local` | Sanitized current state with local tensor references | Same repository only |

Every mode includes an `origin/**` payload. In a referenced-local bundle the exporter emits valid safe-tensor placeholders until import resolves the four declared hashes. Export first stores the real tensor bytes in the repository `.blobs` store. A different installation without those exact objects rejects the import, so referenced-local files must not be advertised as portable or shareable checkpoints.

## Privacy and secret-redaction boundary

For `current-portable`, `origin-portable`, and `referenced-local`, the exporter:

- removes training-source local paths, retained raw text, and source blob hashes from application state;
- removes retained `raw_text` from worker source records;
- removes statements from document/import idea records;
- excludes content-addressed source blobs;
- downgrades every non-`Off` tool grant to `Ask`;
- recursively redacts recognized credential-shaped values in application and engine JSON.

The recursive redactor replaces values under credential-like field names and recognized private-key, AWS access-key, GitHub token, OpenAI-style key, bearer-token, password/assignment, and credential-bearing URL patterns. It runs in private-archive mode as well.

A private archive may retain source paths, raw text, and source blobs only after explicit desktop confirmation. Before packaging, each text-like retained blob is scanned with the same patterns; export refuses the archive if a likely credential is found. Binary blobs are sampled to determine whether they are text-like. This is a deliberate practical boundary, not a proof that arbitrary user-authored or binary data is secret-free. Users must still inspect sensitive archives before sharing them.

The bundle does not contain the app's browser partition, operating-system credentials, or a credential vault.

## Provenance and licensing

The manifest's license ledger preserves the Omni application license declaration plus source name, optional provenance URL, license label, and optional license URL. The state continues to retain source content hashes, import timestamps, policies, and provenance where the selected privacy mode allows it. Import requires a structurally valid ledger, and the desktop can expose those declarations before a pack is installed. License metadata is attribution and warning data; it does not itself verify ownership or grant rights.
