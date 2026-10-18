# Executable training-format parsers

Updated 2026-10-15. In Omni AGI Studio, a plugin may attach either an immutable
import-free WebAssembly record transformer or one exact reviewed Runtime JS
stream session to a training format. Both receive host-owned bounded source
pages and produce verified prepared data before ordinary native learning. They
are not Node/native modules, setup scripts, direct filesystem/network tools, or
replacement brain runtimes.

## Pure WebAssembly declaration and review

`manifest.trainingModules` is optional so canonical packages created before
this feature retain exactly the same JSON and content hash. An executable
package declares:

```json
{
  "capabilities": [{
    "toolId": "data.training-parser",
    "actions": ["sample-lines"],
    "reason": "Execute the reviewed pure parser for selected sources."
  }],
  "trainingModules": [{
    "id": "sample-module",
    "kind": "wasm-stream",
    "abi": "omni-wasm-record-stream-1",
    "wasmBase64": "<canonical base64>",
    "wasmSha256": "<sha256 of decoded bytes>"
  }],
  "trainingFormats": [{
    "id": "sample-lines",
    "title": "Sample lines",
    "description": "Transforms selected records into JSONL.",
    "extensions": ["sample"],
    "mediaTypes": [],
    "parser": {
      "kind": "wasm-stream",
      "moduleId": "sample-module",
      "output": "jsonl"
    }
  }]
}
```

Every module must be referenced. The reserved parser capability must list the
exact executable format IDs, and the current instance must explicitly enable
that capability in addition to enabling the immutable plugin version. Global
disable, instance disable, version change, or capability revocation aborts the
active parser lease. New-owner imports and full merges already keep plugin
access dormant, so imported module bytes never become executable authority.

The package content hash covers the module base64, decoded-byte checksum,
format declaration, capability, documentation, version, and provenance.

## Pure WebAssembly ABI and isolation

The `omni-wasm-record-stream-1` module has no imports and no start function. It
declares exactly one unshared wasm32 memory with both initial and maximum page
counts and exports:

- `memory`;
- `omni_alloc(length: i32) -> i32`;
- `omni_parse(pointer: i32, length: i32, flags: i32) -> i64`;
- optionally `omni_free(pointer: i32, length: i32)`.

`flags & 1` marks the final empty input. The returned `i64` packs the output
pointer in the upper 32 bits and output byte length in the lower 32 bits. Each
nonempty output batch must contain complete newline-terminated UTF-8 JSON
objects. A module buffers any source record split across input chunks in its
own bounded memory and emits it only when complete.

The trusted host compiles and instantiates the module in a killable Worker.
The module receives only copied input bytes; it has no Node object, path,
filesystem, network, credential, process, tool executor, or Studio bridge.
This does not use Node `vm` as a security boundary.

Worker heap limits are not treated as a WebAssembly-memory limit. Before the
Worker starts, Studio parses the module memory section, rejects imported,
shared, memory64, maximum-free, multi-memory, and start-function modules, and
admits the declared maximum linear memory plus module/batch copies against a
physical envelope derived from current free RAM and heap headroom. Every input
and output pointer/length is checked for unsigned overflow and against the
actual current memory buffer after each call.

## Pure WebAssembly streaming, coverage, and recovery

Main—not the module—opens the exact manifest-owned source. One bounded chunk is
in flight at a time. The next chunk is not read until the Worker returns and
the host has validated and durably drained the previous JSONL output, providing
backpressure. Source and output SHA-256 digests, byte counts, and output record
counts are accumulated incrementally. There is no total-source or total-output
record cap; each output write still checks the current disk reserve.

Completion requires traversal of exactly the committed source byte count and
checksum, a successful final ABI call, valid complete JSONL batches, a closed
output stream, and an atomic spool receipt. The receipt binds plugin/package,
format/module/ABI, source checksum/bytes, output checksum/bytes/records, and the
dataset transaction. Reuse rehashes both source and output before trusting the
receipt.

Cancellation or access revocation terminates the Worker. A trap, invalid UTF-8,
partial JSONL record, unsafe JSON key, pointer error, memory pressure, source
change, or disk pressure writes explicit incomplete parser coverage and leaves
the manifest cursor unadvanced. It never reports 100%, skips the remainder, or
commits a neural record. Resume may deterministically reconstruct the spool;
only a verified complete spool enters the existing native JSONL ingestion and
its durable record cursor.

The final training-source ledger keeps the original selected source checksum
and bytes, the immutable parser-binding checksum, and the exact executable
parser provenance. Temporary transformed bytes are removed only after the
native receipt and Electron-side source bookkeeping commit; interrupted work
retains its transactional spool for safe verification/reuse.

## Reviewed Runtime JavaScript parser

A Runtime JS parser uses a module already present in the immutable package's
`runtimeModules` array. Its training format declares one of:

```ts
parser: {
  kind: "runtime-stream";
  moduleId: string;
  output: "jsonl" | {
    kind: "media-assets";
    mediaTypes: Array<
      "image/png" | "image/jpeg" | "image/gif" | "image/webp" |
      "audio/wav" | "audio/x-wav" | "audio/flac" | "audio/mpeg" |
      "video/mp4" | "video/quicktime" | "video/webm"
    >;
  };
}
```

The current instance must enable the exact package/version and the reserved
`data.training-parser` capability for that format. Main starts a fresh
`purpose: "training-parser"` runtime session bound to brain, package hash,
module hash, format, dataset transaction, source hash/bytes, observed input MIME,
and output declaration. It does not reuse or implicitly Activate an ordinary
feature-runtime session and it does not require a loaded neural worker merely
to prepare the source.

Module code registers exactly one parser with:

```js
globalThis.omniRuntime.registerParser({
  push(step) { /* consume the next bounded input page */ },
  drain(step) { /* return the next bounded output page/state */ }
});
```

`push` and `drain` receive exact sequence and output-envelope information. For
JSONL, results are `need-input`, bounded `output` bytes, or `done`. UTF-8 and
records may span output pages; host validation carries incomplete tails across
pages and commits only complete newline-delimited objects after full source
traversal. For media, results are ordered `asset-start`, `asset-chunk`, and
`asset-end` records followed by `done`. Main validates kind/MIME, canonical
base64 pages, offsets, size, checksum, and supported container signature. It
then commits an exact media-only ZIP/ZIP64 archive and asset manifest for the
normal native archive cursor. Raw image/audio/video bytes are not converted to
captions or text lessons. A complete result with zero assets explicitly causes
zero modality updates.

Training-parser sessions cannot mount UI, publish dynamic tools, access general
module state, inspect raw source paths, or receive credentials/approval tokens.
They may use `omniRuntime.call` only for capabilities declared by the package
and enabled for this instance. Every call re-enters the nested tool's current
Off/Ask/Auto/Full policy. Ask appears in the trusted Studio review surface,
including during initial source preparation before Chat opens, while progress
truthfully reports `waiting-for-approval`. Deny/refusal remains a real parser
result; the module cannot approve itself.

Host call identity is derived from the source transaction, module/package,
format, call ordinal, and exact operands. A matching completed effect reuses its
receipt; changed operands reject, and an uncertain dispatched effect is not
silently repeated after interruption. The host streams the complete selected
source with backpressure, rechecks current package/access/source identity,
admits RAM/disk at each boundary, and closes output plus canonical provenance
before handing prepared JSONL/archive data to native ingestion.

## Current limits

Executable preparation does not prove later neural learning, comprehension,
retention, or useful automatic parser selection. Runtime media output currently
supports only the declared image/audio/video MIME list above and its validated
containers; it is not a general binary decoder or arbitrary codec hook. Custom
Node/native modules, setup scripts, direct application/decoder internals,
unreviewed host calls, broader codecs, and unrestricted filesystem/network
access remain unsupported. Packaged cross-platform and hostile-package
qualification and trained natural selection remain separate acceptance work.
