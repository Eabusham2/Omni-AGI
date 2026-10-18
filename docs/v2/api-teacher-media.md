# Typed media lessons from an external teacher

Source contract added October 17, 2026. This extends the existing Teacher card,
job service, authenticated SQLite history and native dataset consumer. It does
not replace OmniCortex with the provider model. A provider is an optional source
of training examples; the same native brain learns the returned media.

## Selection and current documented adapters

Omitting `responseModalities` preserves the existing text-only request and
trajectory path. The existing card offers output checkboxes for the explicit
adapters below. Model selection remains user-controlled. Discovery does not
prove a model/account can generate any selected modality.

- OpenAI Responses: image output through its image-generation tool; actual
  `image_generation_call.result` bytes are admitted, not a description of an
  image. This uses the documented [image generation response contract](https://developers.openai.com/api/docs/guides/image-generation#generate-images).
- Gemini Interactions: typed `model_output` content can carry image, audio or
  video data. The request asks for inline containers (WAV for audio) using
  `response_format`. The source follows the [Interactions API reference](https://ai.google.dev/api/interactions-api),
  including typed content and output-format declarations.
- OpenRouter: the explicit image adapter uses its dedicated `/api/v1/images`
  endpoint and requires identifiable base64 image bytes, following its current
  [Image Generation documentation](https://openrouter.ai/docs/guides/overview/multimodal/image-generation).
  It does not invent image support on the existing chat-completions endpoint.
- Claude and local Ollama retain their text-teacher paths. Unsupported output
  combinations are rejected before dispatch, without changing the model or
  silently requesting captions instead. This is not a claim that those
  providers can never support other media through a future documented adapter.

These official contracts were read on October 17, 2026. No provider account,
credential, network generation, model availability or billing result was tested.

## Real assets, ordered alignment and actual learning

`apiTeacherMediaProviders.ts` normalizes only declared typed outputs.
`apiTeacherMedia.ts` verifies canonical base64, MIME/container signatures and
actual decoded byte hashes. A caption, label, URL string, tool instruction or
missing media part cannot satisfy an image/audio/video request. Incomplete
provider status and additional typed-block instructions are refused.

Accepted assets are immutable content-addressed files under
`api-teachers/assets/`. The authenticated `mediaLesson` preserves original
provider/model/endpoint, delivery identity, output order, literal text parts,
asset MIME, byte counts and hashes. Its deterministic `lesson-<sha>.jsonl`
manifest contains a text trajectory and exact `omni-teacher-media-pair-1`
records. Annotation field ordering is explicit, so SQLite canonical JSON does
not change the manifest hash after a restart or copy.

The native dataset parser recognizes that exact descriptor, verifies its local
relative asset path/hash/container, and yields a real binary `DatasetRecord`.
Large JSONL conditions retain the typed route after spooling; they are not
silently treated as generic text. Malformed typed descriptors do not fall back
to caption learning. Literal verified prompt/annotation text conditions the
existing `_train_media` losses and shared idea space. It is distinct from an
audio transcript: descriptions do not enable speech-pair objectives. Embedded
video audio keeps its existing real-audio path and receives the same verified
condition when present.

`BrainService.learnTeacherMediaExperience()` uses the existing neural writer,
explicit-job preflight and `ingest` transaction. It validates all assets and
the manifest before dispatch. Completion requires exact owner/content/delivery
and transaction identity, native `modality_trained`, complete source coverage,
zero failed media records, and actual per-modality record/window/unit coverage.
A text-only parameter change is not a media-learning receipt. No model is
constructed by importing any of these helpers.

## Response custody, resources and interruption

Complete provider responses stream into hash-bound `envelopes/<sha>.json`
custody before their potentially large JSON/base64 representation is parsed.
`response-staged` means received, not learned. Selected-family admission plus
current physical/V8 headroom covers response parsing, manifest construction,
hash windows and asset publication; disk checks cover writes. Media does not
inherit the older text-response 16 MiB envelope cap. Resource pressure pauses
with received bytes retained, without truncation or a repeated paid request.
Native media decoding/windowing retains its own existing resource checks.

The first implementation is deliberately inline-only. Streaming publication
refuses known device secrets and URL/auth-bearing response envelopes, including
signed handle patterns, before final publication. It does not fetch arbitrary
provider-returned URLs or export authentication headers/session grants. This
does not promise to detect every possible secret or private detail in arbitrary
provider-generated text, pixels or sound. Accepted content remains unsanitized
user data, not a claimed anonymized dataset. Signatures are routing/identity
checks, not a substitute for actual decoding or proof of media quality.

Pause/Stop during an incomplete provider response leaves explicit uncertain
review; it never automatically re-bills. A completely received response that
arrives at the Stop boundary is recorded before disposition. A verified native
ACK is likewise retained if Stop arrives before host receipt publication.
Unknown native acknowledgements remain `learning-uncertain-review`: a matching
content hash alone is not treated as proof. Explicit retry/skip is required.

An unsupported complete response becomes `response-review`. Retrying that
review reparses the already received bytes and does not call the provider.
Parsing/learning an already received lesson can resume after restart even if
its provider key has been removed. New pending provider requests still require
current device credentials. Copied jobs remain dormant and cannot inherit
provider or learning authority from these records.

## Exact snapshot, copy, merge and export custody

`captureApiTeacherWorkspace()` snapshots the SQLite reference set coherently,
then copies its immutable referenced envelopes, manifests and assets. Current
and every exact sealed historical database participate, not only the latest
job cursor. Known but unreferenced interrupted CAS publications are inert and
are not promoted to a learned lesson by namespace alone. Original source IDs,
receipts and bytes survive copy/rekey/merge; live credentials do not.

The existing learning-workspace path now uses that coherent capture rather
than walking a mutating teacher tree. `validateApiTeacherMediaArchiveSources()`
validates exact inventory/blob closure for normal, historical and connected
parent `learning-workspace`/`portable-learning-workspace` scopes. A missing,
changed or unreferenced teacher payload refuses before adoption. Format2 must
declare `api-teacher-typed-media-v1` when this typed custody is present; the root
repository now composes those conditional guards before identity reservation
for ordinary/origin exports, historical-only recovery inventories and opaque
connected parents. Teacher-free archives do not acquire the token globally.

Repository snapshot/clone/recovery/export/import and full-merge
snapshot/adoption/teacher-merge callers forward cancellation-checked selected
metadata admission. SQLite caches and the actual largest JSON-row allocation,
file-hash windows and inline buffers are admitted; this is not a fixed data cap
or another reservation of model memory. Full merge retains hash-sealed teacher
archive databases byte-for-byte instead of rewriting them through SQLite backup.
External archive adoption retires copied teacher authority even when the source
ID is unused; explicit same-owner recovery preserves its selected paused prefix.

## Recorded evidence boundary

Fifteen distinct new host/file/SQLite/fake-provider cases passed across scoped,
affected runs. One new static existing-card presentation case passed. Three
native parser cases passed, including a condition beyond the old 64 KiB inline
text boundary. Real files prove raw byte/hash custody, dormant copy/rekey/merge,
portable inventory refusal, Stop/lost-ACK handling, no re-billing on parsing
resume, removed-key resume, linked-directory refusal, an escaped secret spanning
response chunks, and the actual host
ingestion-method seam. Provider and native replies in host cases are synthetic.

Two actual implementation defects found during those runs were fixed: a path
helper name was shadowed by its argument, and canonical stored JSON reordered
annotation fields in the generated manifest. Only affected cases were repeated.
Existing text-teacher, archive11 and repository51 suites were not rerun.

Five additional composition cases pass across scoped selections: actual
repository conditional-feature/current/origin import, historical-only
namespace/tamper refusal, selected-RAM/late-cancel refusal before identity
adoption, opaque origin/history/connected inventories, and actual inert
full-merge seal/snapshot/adopt/merge callback and exact sealed-byte custody.
They exposed and fixed same-ID external imports retaining resumable teacher
jobs; the test was not weakened to accept that behavior. The full-merge case
uses the actual host sealer/coordinator with a metadata-only assembler, never a
native model. Only affected cases were repeated; the closed19-case slice and
old51/11-case slices were not rerun.

No real provider, key/account, model, Load, training, media decoder/FFmpeg,
neural-quality or hardware-performance evaluation ran. The full trained-brain
test remains user-deferred and is not claimed passed here.
