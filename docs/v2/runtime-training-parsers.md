# Runtime training parsers and pre-chat review

Internal contract and evidence note, October 15, 2026. The parser source owner
has reported final source freeze for JSONL and ordered mixed image/audio/video
output, including fail-visible cached receipts and a live origin guard for host
work which continues after `execute()` returns. This note records those scoped
boundaries, not completion of all R243, a trained capability, or packaged parser
acceptance. The documentation lane inspected source but reran no tests, parser,
model, provider, package, or host effect. Public documentation remains unchanged.

## Scope and source owners

The runtime-stream branch uses the existing isolated JavaScript plugin runtime;
it does not replace Omni's native brain or the pure WebAssembly parser branch.

| Format declaration | Execution boundary | Current output |
| --- | --- | --- |
| `builtin-stream` | Existing selected builtin parser | That builtin's supported format |
| `wasm-stream` | Import-free, bounded WASM transformer; no host-tool bridge | JSONL |
| `runtime-stream` | Exact reviewed `runtimeModules` body in a dedicated sandbox renderer; declared host tools remain permissioned | JSONL, or typed image/audio/video assets in a media-only native archive |

`declaration-only` does not execute a transformer. The pure WASM contract remains
in [executable-training-parsers.md](executable-training-parsers.md); that older
document's broader remaining-scope paragraph predates this runtime-stream slice.
It is not being rewritten or taught as part of this internal note.

Relevant source is `src/shared/{plugins,pluginRuntime,types}.ts`,
`src/main/{pluginCatalog,pluginService,pluginRuntimeBootstrap,pluginRuntimeService,
pluginRuntimeStreamParser,pluginRuntimeMediaSpool}.ts`, and the parser-provenance validation in
`dataIngestion.ts` and `brainActivityLedger.ts`. Main composition supplies
`admitTraining` and `plugins.setRuntimeStreamParser(...)`. Native source ingestion
and its receipts remain separate in `brainService.ts`.

## Exact selection and admission

A training-format declaration can reference an immutable JavaScript runtime
module using this **fragment**, not a complete package:

```json
{ "kind": "runtime-stream", "moduleId": "records", "output": "jsonl" }
```

The alternative mixed-modality declaration is also hash-bound package data:

```json
{
  "kind": "runtime-stream",
  "moduleId": "media",
  "output": {
    "kind": "media-assets",
    "mediaTypes": ["image/png", "audio/wav", "video/mp4"]
  }
}
```

The current admitted MIME set is PNG, JPEG, GIF and WebP images; WAV, FLAC and
MP3 audio; and MP4, QuickTime and WebM video. This is an explicit supported set,
not a claim that every possible codec/container already works.

The module is in `manifest.runtimeModules`, not `trainingModules`. The reserved
`data.training-parser` capability must permit the exact format ID, and the
current instance must enable the exact package hash and capability. The lease
binds the package, format, runtime-module hash, observed normalized MIME where
present, catalog revision, and instance revision. Automatic reviewed-extension
selection and explicit MIME/builtin overrides retain their existing distinct
selection rules; an arbitrary MIME label is not execution authority.

`PluginRuntimeTrainingParserActivationRequest` additionally binds `brainId`,
`transactionKey`, original `sourceSha256`/`sourceBytes`, and request identity.
`activateTrainingParser` and `parserStep` are main-owned host methods, not public
renderer privileges or an arbitrary source-path API. The module receives copied
source bytes and identity metadata, not the main source path or workspace root.

Parser sessions have `purpose: "training-parser"`. Their resource admission uses
the distinct `admitTraining` hook; production composition admits host/runtime
resources without loading a neural worker merely to transform or review data.
The renderer must acknowledge ordinary startup and register the exact parser
format/ABI before activation. A missing or failed readiness acknowledgement is
a bounded runtime error, not a model quiz or a readiness test of intelligence.

## Record-stream SDK boundary

The ABI is `omni-runtime-record-stream-1`. The following is a signature sketch
of the landed bootstrap API, not an executed parser example:

```ts
type ParserHandler = (
  step: Readonly<PluginRuntimeParserStep>, context: { requestId: string }
) => PluginRuntimeParserStepResult | Promise<PluginRuntimeParserStepResult>;

interface ParserHandlers { push: ParserHandler; drain: ParserHandler; }
// The supplied SDK accepts: await omniRuntime.registerParser(handlers).
// This sketch omits the package's actual ParserHandlers implementation.
```

Register once during module declaration. Registration verifies parser purpose,
format and ABI; effects cannot run during startup. Main subsequently sends:

- `push`: exact `id`, `sequence`, `maxOutputBytes`, canonical `inputBase64`, and
  `final`. Main sends the final flush as an empty input with `final: true`.
- `drain`: the same typed boundary, without another input page or final flag.
- JSONL result: `{state: "need-input"}`, `{state: "output", outputBase64}`,
  or `{state: "done"}`. Only `output` may contain JSONL transport bytes.
- Media result: typed `asset-start` (ID, kind, MIME), ordered `asset-chunk`
  (ID, offset, canonical base64 bytes), `asset-end` (ID, byte count, SHA-256),
  `need-input`, or `done`. A media session cannot return JSONL `output` pages.

For each source page, all returned output is written before another source page
is read. A nonfinal page must eventually request input; the final flush must
eventually finish. Main rejects malformed/noncanonical base64, wrong transitions,
changed operands under a repeated step ID, or a page beyond the admitted frame.
Base64 here is byte transport, not a text lesson or a substitute for sensory input.

The current source page is at most 256 KiB and may be smaller under the admitted
frame. Output pages are independently frame-bounded. Complete JSONL records
**may span output pages**; the adapter scans the completed spool and physically
admits each full record before parsing it. Its 64 KiB scan buffer is not a
64 KiB record limit. Per-record admission currently requests
`recordBytes * 12 + 65,536` bytes, with safe-arithmetic checks; this estimate is
not an OS-enforced total-memory cap or proof that an arbitrary large record fits.

Each record must be a newline-terminated, nonempty UTF-8 JSON object. Validation
rejects invalid UTF-8/JSON, unfinished final records, unsafe property keys,
non-finite data and excessive nesting (depth guard 64). This is a structural
record check, not proof of semantic correctness or successful native learning.

## Typed media output and native handoff

One exact source transaction may emit an ordered mixture of image, audio and
video assets. Only one asset is open at a time. Main owns canonical ordinal
member names, temporary paths and the final archive; JavaScript never receives
a host path. Every chunk is frame-bounded, backpressured, cancellable and checked
against the shared RAM and disk envelopes. Closure requires the observed offset,
byte count and SHA-256, exact declared kind/MIME, and a supported container
signature before the `.part` file becomes a committed asset.

The app writes a stored streaming ZIP, upgrading to ZIP64 fields when physical
ZIP limits require them. The archive contains media members only—no JSON caption,
base64 lesson or manifest presented as sensory data. Before publication, main
extracts the owned bundle into temporary custody and verifies every member name,
size and hash against the admitted descriptor set. Compact parser provenance
retains source/package/module identity, bundle hash/bytes, asset count,
per-modality counts and the canonical asset-manifest hash. The full canonical
asset closure remains in the transaction receipt used for exact cache reuse.

The prepared source is handed to the existing native `archive` parser. Its
durable record cursor visits each member in order, `_iter_binary_member` emits a
real image/audio/video record with a leased raw-byte file, and ordinary native
modality training applies that record only when the user actually starts
learning. Initial parser preparation and pre-chat review still do not load a
neural worker. If the parser emits zero assets, the host records complete source
traversal with `outcome: "no-assets"`, zero records/parameters/modalities and no
worker request; it never fabricates a learned source from empty output.

## Pre-chat host-action review

An optional reviewed parser can prepare a selected source before first Chat is
available. Its ability to transform bytes does not grant file, network, process,
credential or other host permissions. `omniRuntime.call(...)` still requires
the package's declared enabled capability and the current ToolExecutor policy,
including Off/Ask handling, exact operands, resource checks and durable receipts.
The sandbox itself has no Node, direct filesystem, direct network or key access.

Training-parser sessions cannot publish general dynamic tools, mount Studio UI,
consume ambient host events, or use persistent `state.get/read/put`. Mutable
parser state stays in the fresh session. These restrictions do not disable the
permissioned host calls explicitly allowed for this parser.

When a permitted call returns a detached host job, the runtime supplies a
main-only `HostActionOriginAuthority`. Its opaque revision binds brain, plugin,
package, module, live session, catalog/instance revisions, tool, action and the
actual argument hash. `assertCurrent()` rechecks the live runtime, exact enabled
capability and abort signal after asynchronous boundaries. The closure, signal,
approval tokens and raw arguments are never written to runtime history or a
portable artifact. Reviewed retries receive the same boundary. A stopped module,
changed package/access revision, revoked capability, changed operands or aborted
owner makes the guard reject. Each detached consumer must invoke that guard at
its own effect/publication boundaries. This supplements, and never replaces,
ToolExecutor's current policy and user review; the two fake cases do not prove a
real detached mutation consumer.

`PluginRuntimeParserProgress` carries exact parser/source identity, byte and
record counts, state and message. `waiting-for-approval` can carry safe
`sessionId`/`rpcId`/`reviewId`/`executionId`, tool ID and action. It never gives
the child or progress consumer the private approval token. The trusted approval
surface shows full operands and reviews the actual current request; approval
does not broadly grant subsequent actions. Deny withholds that action. Whether
the parser can recover from the failed call depends on its handler; no success
is fabricated and Deny is not promised to cancel every possible parser.

The initial-Build and resumed-initialization surfaces mount the shared approval
queue without requiring model output or a hidden Load. Progress callbacks are
serialized, including approval transitions. Parser completion occupies the
preparation phase (fraction 0.25 in the current adapter), **not 100% native
training**, comprehension, retention or completion of the overall Build.

The separate [training-review-ui.md](training-review-ui.md) records root's actual
Electron presentation check with synthetic transport: light and 360px dark
content, independent Deny, expanded complete operands/IDs, bottom controls,
visible focus and Tab/Return approval. It is not an actual parser/initial-training
or real host-effect acceptance result.

## Source custody, publication and retry

Main opens the manifest-owned regular source and reads through the verified open
FD. Where supported it uses `O_NOFOLLOW`; opened-FD and current-path identity,
size and cancellation are checked around reads. Full byte count and SHA-256 must
match the selected source. Receipt/output hashing and reads use guarded regular
files rather than trusting a pathname alone. Windows does not gain a fictitious
`O_NOFOLLOW` guarantee; the identity rechecks remain part of that path.

The app-owned `plugin-runtime-parser` directory contains transaction-named JSONL
or media receipts. JSONL writes begin at an exclusive random `.incomplete` path.
Media writes use an exact transaction directory with ordinal per-asset `.part`
files and `media-assets.zip`. Output errors are surfaced across step/write/end
boundaries and all owned partial custody is cleaned. Completion requires full
source traversal, format-specific validation, exact committed output closure,
isolated receipt write and byte-for-byte receipt readback. Activated session,
owned output and approval subscription are retired under cleanup, including
failure before the first useful parser result.

The `omni-runtime-parser-spool` receipt (schema 1) binds the transaction and an
`omni-plugin-parser-execution` provenance record (schema 1): package/module/format/
ABI, original source hash/bytes/MIME, transformed output hash/bytes/record count,
optional exact media closure, and `complete: true`. Reuse verifies exact canonical
identity and rehashes source/output; media reuse also reopens and verifies every
archive member. **Only a missing receipt (`ENOENT`) is treated as no cache.** Unsafe,
corrupt, mismatched or admission-refused existing metadata fails visibly; it is
not silently discarded to rerun a potentially effectful parser.

Fresh same-transaction reconstruction uses transaction-stable host-call IDs.
`pluginRuntimeHostRequestId` binds brain, package/module hash, format, transaction,
source hash and the bootstrap-owned call ordinal, not a fresh session UUID.
Parser code cannot choose a replacement request ID. Matching completed
ToolExecutor receipts can be reused; changed operands fail, and an uncertain
already-dispatched effect is not blindly repeated. Completed step receipts are
distinct from mutable in-session parser state; a dispatched step without a
terminal receipt is not replayed into that same state. These boundaries are not
a universal exactly-once guarantee for arbitrary outside systems.

Only verified JSONL or a verified media-only archive is handed to ordinary
native ingestion. Its normal record cursor, neural/source receipt and Electron activity/source ledger
must complete before final transformed-spool cleanup. Parser provenance remains
attributable to the original selected source; it is not itself a neural receipt.
Stop/revocation can prevent remaining work and clean partial owned output, but
cannot undo already-completed host effects or already-committed neural records.
Imported/copied module bytes and historical receipts are not live permissions.

## Scoped evidence as reported by the implementation owner

This documentation lane did not rerun these checks. Counts below describe the
reported runs; an affected case is not an additional distinct case.

| Reported phase | Result and boundary |
| --- | --- |
| Original adapter boundary | 3/3 focused fixtures; tiny host/protocol data, no model |
| Cleanup and source replacement | 2/2 targeted cases |
| Record spanning output pages | 1/1 targeted case; not large native ingestion |
| Serialized approval transition | 1/1 affected case |
| Interrupted transaction-stable host call | 1/1 focused fake retained-effect case; not a real external effect |
| Early asynchronous output failure | 1/1 new targeted case, 49 ms; prior six skipped |
| Mixed typed media custody and native archive handoff | 6/6 focused source/fake-host cases across `pluginRuntimeMediaSpool.test.ts` and `pluginRuntimeMediaParser.test.ts`; mixed image/audio/video, exact cache reuse and zero-assets behavior, no neural worker |
| Detached runtime-origin authority | 2/2 focused fake queued-context cases in `pluginRuntimeOriginAuthority.test.ts`; capability/revision/operand/Stop/abort and post-await revocation, no real detached effect |
| Final source compiler/whitespace checks | Owner reported Node/Web TypeScript and scoped diff check clean; root also reported coordinated Node/Web compilation clean with picker/manager/main wiring |
| Owned Electron JSONL parser | Approved affected retry passed in 2,039.119 ms app time: 8,193 source bytes (`10a7c283…df73`), 5 pushes/8 drains, 11,253 output bytes/5 genuine JSONL records (`85bb9c8b…d85b`), one synthetic exact Ask/review, zero brains/neural/provider/real effects and zero live WebContents. Package `7fdcf278…aa9`, module `b14f868b…a2f`; full source points and artifact hashes are in [runtime-parser-owned-electron-evidence.md](runtime-parser-owned-electron-evidence.md). The preserved original admission refusal remains failed. |

The owned Electron retry predates the mixed-media and origin-authority source
additions. It proves the retained JSONL path only at the recorded source/build
hash point; it is not live evidence for the later media or detached-job code.

The source owner confirmed the final guarded receipt semantics at freeze. Its
focused fixtures are in `tests/pluginRuntimeStreamParser.test.ts` and
`tests/pluginRuntimeParserEffects.test.ts`; the new media and origin-guard
selectors are `tests/pluginRuntimeMediaSpool.test.ts`,
`tests/pluginRuntimeMediaParser.test.ts`, and
`tests/pluginRuntimeOriginAuthority.test.ts`. Root's separate synthetic approval
presentation evidence is linked above, rather than counted as a parser execution.

## Remaining acceptance and unsupported output

- The bounded owned Electron JSONL parser path is proven only with generated
  bytes, a synthetic catalog/review and no real effect or neural worker. Actual
  initial-Build learning, hard Stop at each phase, packages and other OSes remain.
- Typed binary media custody is source/fake-host tested only. No actual native
  image/audio/video decoder, modality update, checkpoint/restart or learned
  quality was run. Pre-custody checks classify declared MIME/kind and container
  signatures; full decode acceptance still belongs to ordinary native ingestion.
- The admitted MIME set above is intentionally finite. Other formats need an
  explicit validator/extension/native decoder mapping rather than being called
  universally supported.
- The split-record fixture establishes page-boundary handling, not a genuinely
  large-record full native ingestion or every resource-pressure envelope.
- Real host-effect interruption, hard-crash/lost-ACK recovery, and copied/merged
  pending-authority adoption need their own full-chain evidence. The stable-ID
  fake effect case does not certify every ToolExecutor adapter or remote service.
- Initial Build through pre-chat review, actual parser, native learning and
  source receipt has not run end to end. Structural records and clean compilers
  do not establish understanding, parameter-learning quality or recall quality.
- Packaged platform/hostile-package qualification and whole-app overlay stacking
  remain separate from the scoped source and synthetic component checks.
- New brains, corpus training and the disposable merged-native resume fixture
  remain explicitly deferred by the user; this note does not begin those tasks
  or relabel that deferral as an implementation failure.

## Deferred public documentation draft and teaching plan

No public file or pack is changed here. Once the source/acceptance boundary is
approved for publication, the proposed edits are:

1. `plugins.md`: document the separate runtime-stream declaration, exact module/
   format capability, `registerParser` push/drain signatures, nested host policy,
   guarded source identity, JSONL pages and typed media assets. Preserve the pure
   WASM no-tools boundary rather than weakening it to match runtime modules.
2. `USER_GUIDE.md`: explain optional source preparation and exact host review
   before chat; distinguish parser byte coverage from native learning. Do not
   promise automatic grants, a blocking tool quiz or implicit model Load.
3. `INTEGRATIONS.md`: distinguish main-only activation/step methods from trusted
   review APIs, their identity/progress/provenance DTOs and later native receipts.
4. `executable-training-parsers.md`: add a separate runtime-stream section and
   update only the now-outdated runtime-parser remaining-scope statements while
   retaining the actual WASM ABI and its own evidence limits.

Candidate public user wording, **not yet published or taught**:

> A reviewed runtime parser can transform a selected source into JSONL records
> or typed native image/audio/video assets.
> Any host action still follows the instance's current permissions and may ask
> for approval during source preparation, even before chat opens. Parser progress
> reports source coverage; the normal learning process runs afterward.

Candidate SDK wording, **not yet published or taught**:

> Register one `push` and one `drain` handler for the exact training-parser
> session. Consume the supplied byte pages, return bounded output pages, and
> finish after the final empty push. JSONL uses output pages; media uses ordered
> asset start/chunk/end messages and never substitutes captions for raw bytes.
> Host calls use the declared capability and
> current policy; parser sessions cannot create UI, publish tools or persist
> general runtime state.

No documentation-pack version is reserved by this internal note. A later stable
public update must retain the existing public record set, recalculate exact
content/pack hashes, and add its own declaration check. No teaching runs now,
and internal bug history or synthetic evidence claims must not be baked into
user-facing teaching prose.
