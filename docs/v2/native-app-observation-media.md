# Ordinary App sensory learning and current-ACK audit

This extends the text admission boundary in `native-app-observation.md`.
The source uses the same brain; no foreign model, generated caption, textual
base64 representation, hidden persona, selector optimizer or corpus training
is introduced. Broad trained capability remains a separate acceptance task.

## Actual path

Every reviewed block retains its original position and full record/data hash.
Text and inline text resources use the existing VSA activity path. Supported
inline image/audio bodies—including typed resource blobs—use the bounded
decoders in `mcp_sampling_media.py`, actual
`modalities.perception_embedding`, and the brain's ternary sensory lift. GIF
frames and complete audio samples traverse admitted windows; there is no
ten-second audio truncation. MIME/container/signature, decoded pixel/forward
scratch, RAM-first/spill/codec and cancellation checks remain in force.

Ordered windows enter one compensated, position-bound VSA accumulator. After
all sources complete, one real `_admit_sensory_embedding` applies packed
neural assembly/contact learning, STDP, working activity, automatic settling
and replay. Connected regional observation and root publication share a
`ConnectedExperienceTransaction`, releasing derived-store locks with
`prepare_publication` before the same atomic root save. Root/regional failure
fences the warm owner; it cannot resume from partially admitted activity.

Resource links encode only their literal reviewed reference metadata and are
marked `fetched=false`. No linked contents are implicitly fetched. Unsupported
binary bodies reject explicitly instead of being dropped or called learned.
Mixed/media-only App chat follow-ups remain explicitly unsupported: an already
applied observation/audit must stay visible even if no follow-up can run.

## Durable custody and cancellation

The native head has an optional constant-size `content.summary` and
`content.audit` descriptor. Pure text head hashes stay unchanged. Detailed
ordered receipts stream to one content-addressed JSONL ACK file, with source
SHA/byte/window/sample coverage, never raw media or captions. The file is
finalized before mutation and referenced by the same atomic head publication.

Main verifies/copies that raw audit into its immutable store before host ACK.
Every next claim, including a new stream or superseded record, must provide
the exact `previousAuditArchivedSha256`. Missing/mismatched archival proof
backpressures before decoding, learning or cleanup. An exact duplicate returns
the existing audit without decoder or learning execution. Only after the next
root commit may the exact verified previous file be removed.

Native snapshot sizing/copy and cold Load preserve/verify the active audit;
main owns corresponding fork/export/import/recovery/full-merge closure hooks.
Original source ownership is not rewritten on copies. Missing or corrupt
current files fail integrity, not as permission to relearn an applied record.

Reading is cooperatively cancellable through the exact active request and
temporary activity pager. Pre-mutation cancellation returns `-32800` with
`learningApplied=false` and leaves the owner resident. Once mutation starts,
atomic publication settles before its receipt is returned. Unknown publication
returns `committed=null`, not a claim that nothing happened; its possible ACK
file is retained. Confirmed publication with a missing audit reports applied
state requiring audit repair. Both fence/release the warm owner; no implicit
Load or automatic replay follows.

## Evidence

- Ten constructor-free tests passed in 1.539 s: real small PNG/WAV codecs with
  explicitly synthetic encoders/admission, complete mixed ordering, one
  synthetic admission, current-ACK retry, archival backpressure, copy/recovery,
  resource/cancel/invalid signal, save failures, tamper and connected call order.
- The fixed content-head UTF-8 hash agrees across Python and TypeScript.
  All four TypeScript hash cases passed in 9 ms; 15 affected text/source cases
  passed in 0.139 s.
- Three additional affected no-model cases passed in 0.150 s: correlated
  serial read cancellation, unknown publication/audit preservation and applied
  publication with missing audit. Python compilation and diff checks passed.
- Main's raw-source/archive/portability/late-commit fixtures are separately
  recorded in `mcp-interactions-apps.md`; they do not substitute for native
  packed sensory mutation evidence.

### One actual native attempt: committed state, failed full assertion

The one authorized attempt finished in **22.183 s**, at a measured peak process-
family RSS of **240,205,824 bytes** (about 229 MiB). The watchdog limits were
30 seconds and 1.5 GiB; neither limit fired. It loaded one isolated copy of
the previously preserved tiny native checkpoint (9.246 s Load), read one
8-by-8 PNG and a 137-sample WAV, and executed real native sensory admission.
The protocol/review envelope was an explicitly synthetic, owned fixture—not
a genuine MCP server call or UI approval. No user corpus, decoder generation,
optimizer step or second model reload ran.

The adapter returned `reloadRequired=true`, so the proof failed its first
post-admission assertion. The later live duplicate/no-decode assertion was
**not reached**. This is not recorded as a fully passing native media test.
Its phase log and preserved fixture are named in
`work/v2-native-app-media-proof-2026-10-11/attempt.jsonl`; that log also prevents
automatic reruns.

A separate **file-only** inspection, using only Python's standard library
and no native/runtime/model imports, verified the committed output against
the real before-state and raw approved fixture bytes:

- Packed neuron rows grew **7 to 21**, packed row bytes **28 to 84**, and
  connections **42 to 204**. All referenced immutable shard hashes verified.
  All seven prior vector rows retain identical signs, contact state, counters
  and repeated-contact bytes. The measured vector mutation is **14 new packed
  rows**, not evidence that pre-existing vector levels were rewritten.
- Packed signs SHA-256 changed from
  `ae3e76858bc470330722f561fbfd671521cad8f561e68f90d059c9938397fe30`
  to `5f55b14585265440f14162f682ce64b259f80b2237b440ab33a846adc2521dfd`.
  Combined sorted packed/contact/counter bytes changed from
  `5bb65fd597e9aac3e7061e9f02f62bd96c81f702a096eec69c4e7bee2e24331d`
  to `13e9eb5a4a0299fcdef11b27a4fda78859188a7878313c647f5994251a436a59`.
- The atomic ordinal-2 head and its 1,586-byte/two-row audit agree with both
  complete raw media SHA/byte counts and ordered block digest. Saved coverage
  is one image frame, all 137 audio samples, four decoded windows and
  `wholeInputTraversed=true`. Experience count advanced once, **1 to 2**.
- `training_steps=0` and `inference_count=0`. The full core blob remained
  exactly `71f2a05f4af4fa67453612e67604eeac9866cf298e03bf86d9680c6a294d007a`;
  plasticity blob changed. This proves packed sensory-memory mutation, **not**
  a slow language-weight update or trained perceptual understanding.
- The three original checkpoint files are unchanged. The previous text head
  was continued by its real predecessor; it was not retroactively given a
  fabricated audit or source seal.

The likely failure cause is an overly broad proof spy: it forbade
`conversation.recent_payloads`, while the production save calls that method
after atomic publication to refresh bounded UI/history caches. That is not
prompt injection. The original caught cleanup exception was not retained, so
the cause remains an **inference**, not a definitively observed attribution.
Product history-cache behavior was not changed. Future proof instrumentation
permits only the exact post-save cache refresh while still forbidding chat,
generation, tokenization and prompt/source lookups; the corrected proof has
**not been executed**, and no retry is authorized here.

### Current source classification · 2026-10-13

A read-only review compared the current adapter files with the SHA-256 values
recorded by that original attempt. All three are **exact matches**:

| Module under `engine/omni_core/` | Original attempt and current SHA-256 |
| --- | --- |
| `app_observation.py` | `77d24cc2202d9e7ad07f1de6e49399836efa438d5f0a96e991c4393a309bd2f4` |
| `app_observation_content.py` | `7a4e64f6eb32844e87fb88aec23fbf4b1c26f3d70db3b4cac3919d597c9b78bc` |
| `app_observation_audit.py` | `07b271b2a822114af5ed07853836e079866e603fbd6a9778b2a5091442240b85` |

Consequently this is not labelled an App-media bug silently corrected by later
packed-state or Load work. No current adapter/persistence wiring defect was
confirmed from this record. `AdaptiveBrain._save_checkpoint_impl` still
refreshes `conversation.recent_payloads` only after atomic `brain.json`
publication and its commit callback. A blanket assertion against that read
would enter the intended confirmed-publication-but-fenced-owner branch of
`admit_native_app_observation`. This explains how the retained committed
sensory bytes and `reloadRequired=true` can coexist; it remains an **inference
about the old failure**, because the original caught exception was not saved.
It is not retroactive proof that this was the actual exception.

The current host/worker wiring was also inspected without executing a brain:

- `Worker._get` defaults to `allow_load=False`; the App bridge does not opt into
  Load. A fenced/missing owner cannot be implicitly recreated by an observation.
- `EngineSupervisor.loadedAppObservationRequest` uses the existing serial
  loaded child, validates it again after queue admission, and never starts or
  replaces a worker for this operation.
- `McpAppObservationDelivery` archives and validates the current detailed audit
  before host ACK. A successful late published receipt is retained even when
  its warm native owner must be released. Both typed release results and
  correlated `-32031` errors invalidate that exact owner's host admission;
  the index composition holds further dispatch and suspends its MCP handles.
  No implicit Load, automatic relearning or follow-up proceeds through that
  released owner.

The published-but-fenced recovery behavior and normal post-save history cache
were deliberately **not changed** to make the proof pass. No old native proof,
old constructor-free suite, model initialization/Load, inference, training or
provider call was rerun in this review. The prior full assertion remains failed,
and live duplicate, genuine App approval-to-native integration and full-chain
media/quality acceptance remain open exactly as recorded below.

### Separate approved host-chain attempt · 2026-10-13

One later, explicitly approved private-copy attempt is recorded under
`work/v2-native-app-media-retry-2026-10-13/`. It stopped after **4.429355 s**,
before `admit_app_observation`. Actual Load completed in about **2.190 s**;
the real host frontier reader then rejected the already-retained native head's
`streamId="app-stream"` because it required a UUID. This run did not reach
media decoding/learning, native save, sequential reload or duplicate. Its
pending host queue is not an admission receipt.

Peak sampled watchdog-plus-owned-family RSS was **387,022,848 bytes** (about
369 MiB), private allocated disk **1,458,176 bytes**, and minimum observed free
disk **34,214,268,928 bytes**. No resource stop fired. All 75 original closure
files remained unchanged. Native shutdown recorded zero remaining brains;
an independent process check found no owned watchdog/host/native child.
Its watchdog log and private copy remain untouched; the one-attempt `open('x')`
guard is not cleared to silently rerun it.

This exposed a **separate current host/native shape mismatch**, not the old
post-publication failure. Current `app_observation.parse_app_observation` and
`validate_frontier_head` require the ASCII opaque identifier grammar
`[A-Za-z0-9][A-Za-z0-9._-]{0,127}` with true full-match ending. Admission stores
the supplied `record.streamId` unchanged; it is part of the native frontier
digest, exact FIFO predecessor and current-head duplicate identity. The shared
head type is an opaque string. Native validation never imposed a UUID here.

The main App queue **still produces UUID stream IDs** and validates UUID
artifact/observation/review/queue identities. Those contracts were not changed.
An App observation stream is also not an inherited chat's `rootTurnId`:
the current native continuation producer creates `ancestral-` plus a digest,
but it does not automatically assign that value to App streams. The actual
retained originD9 ID
`ancestral-50bcb3baad27d8fdee107612a3fb17cf896769d9ac20dd2226b69e8619b6de5b`
is used only as a native identifier-grammar fixture, not a fabricated App
producer or admitted observation.

Only `checkedAppObservationFrontier` now requires a **string** matching that
current native grammar and true EOF. Its JavaScript guard does not use `$`
alone, which could accept a trailing newline. All exact native head fields,
brain ownership, digest verification, reviewed record matching, predecessor
and expected-frontier/CAS checks remain intact. Neither saved head nor source
identity was rewritten to make the fixture pass.

Five new constructor-free cases passed in **4 ms** in
`tests/mcpAppObservationNativeIds.test.ts`: the actual retained opaque head/hash;
the actual inherited-ID grammar example; rejection of newline/CR/NUL/numeric/
oversized/invalid-first/Unicode IDs; unchanged owner/hash/record/CAS rejection;
and unchanged queued UUID requirements. No old suite, neural constructor,
Load or actual retry followed this source correction. These cases establish
the contract fix only; the full host-to-native media boundary remains unproved.

The file-only verifier is `inspect_saved.py` beside the phase log. The earlier
text mechanism proof and later timed-out reload strengthening remain distinct.
Native live duplicate after this actual media admission, genuine App-host/UI
approval-to-native integration, mixed/media-only App chat follow-up, model
quality and packaged-device coverage remain open; synthetic contract cases
do not establish them.

### Separately approved affected structural PASS · 2026-10-13

After root reviewed the native-ID correction and five new constructor-free
cases, ONE newly authorized attempt used a separate explicit log and private
copy. The command was:

```sh
./.venv/bin/python work/v2-native-app-media-retry-opaque-id-2026-10-13/watchdog.py --approved-structural-attempt
```

It **passed in 8.425648 s**, exit0, with no resource cap firing. This was actual
production `McpAppArtifactStore` / `McpAppObservationDelivery` / serial writer
calling the actual `Worker` on one copied existing native checkpoint. No new
brain, optimizer, generation, corpus, provider call or changed saved head was
used. The reviewed task was a root-approved task-owned protocol fixture, not
a genuine external MCP completion or human App review click.

Durable receipts preceded the proof assertions. The real media admission
returned an observed native head with no owner-release requirement; main
verified and archived the detailed audit before durable host ACK, leaving
admitted1/pending0. The run then genuinely unloaded this owner, verified both
old brain and ModalityHub weakrefs dead with no resident brain, and reloaded
the SAME private committed copy. An exact duplicate returned the original
head without another sensory call, mutation or save.

| Measured clause | Before | After admission / reload / duplicate |
| --- | --- | --- |
| Packed neuron rows / packed row bytes | 7 / 28 | 21 / 84, preserved across reload and duplicate |
| Ordinary experiences | 1 | 2, still 2 after duplicate |
| Plasticity events | 1 | 19, unchanged by duplicate |
| Training steps / inference count | 0 / 0 | 0 / 0 throughout |
| Actual sensory encoder windows | 0 | 4 during admission; 0 after reload/duplicate |

The measured all-packed-state SHA changed from
`5bb65fd597e9aac3e7061e9f02f62bd96c81f702a096eec69c4e7bee2e24331d`
to `937313d3c12bbb67b47537247a7b791f25e1f42931895573aea8116769a18f37`;
that exact latter SHA survived actual unload/reload and duplicate. The core
parameter checksum stayed
`f4e60f50b96eb533002b417ddb9ead547c997cfe4fa2fbf32511fe2cc47673de`.
Raw token context remained empty and the pre-existing pending slow-learning
queue was unchanged. The strict optimizer/chat/generation/tokenization/source
retrieval guards recorded no calls. Normal exact-save UI history refresh was
permitted; no actual save error was recorded.

Actual complete sensory coverage was one 8×8 PNG (77 raw bytes) plus the entire
137-sample WAV (318 raw bytes): **395 raw bytes, 4 decoded windows, 1 image
frame, 137 model audio samples**. The source-bound ordered audit contained
**2 rows / 1,586 bytes**, SHA
`1f9b774b424849ff9038ea7396d659770265e197eeca717ad0912a78a24da822`.
This is real same-brain packed sensory/working-memory mutation and persistence,
not a textual caption/base64 substitute or evidence of language-weight training.

Peak measured sampled watchdog-plus-owned-family RSS was **400,670,720 bytes**
(about 382 MiB), private allocated disk **1,912,832 bytes**, and minimum observed
free disk **34,199,736,320 bytes**. RSS sampling is not an OS-enforced hard
limit. All 75 original closure files remained unchanged. Actual shutdown
recorded zero resident brains and unchanged imported native source files;
an independent complete process listing found no owned watchdog, host or
native child. The exclusive neural slot was released and this lane frozen.

Evidence is preserved at
`/var/folders/l1/3bjvcv150hv_srvsrj66t8jr0000gn/T/omni-app-media-opaque-retry-7n1nqlup`
(`host-receipts.jsonl`, `native-receipts.jsonl` and source/custody files), and
`work/v2-native-app-media-retry-opaque-id-2026-10-13/watchdog.jsonl`, SHA
`cb7a305301a572740996a20f433ab20c3cc2936d375b9d24bdc8977831179f1b`.
The original failed 4.429355s log still has SHA
`4e54e7b776849e71da725e4946071523baaf36d222f75f51aa2912919904614b`.
Neither that attempt nor the earlier 22.183s failed full assertion is rewritten
as passing; this separate affected attempt supplies the previously missing
bounded host/native save-reload/duplicate structural evidence.

Root's coordinated current Node TypeScript check passed. Ten pure CI contracts,
including the native-ID contract case, are now wired; this entry does not
claim a new remote CI run. **Perception or language quality, generation, full
semantics, genuine provider→App UI review→native→chat, mixed/media-only App chat
follow-ups, and packaged-device acceptance remain open.** No second retry,
test suite, build or production source edit followed this pass.
