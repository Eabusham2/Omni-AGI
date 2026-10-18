# Same-brain MCP sampling

## Implemented boundary

`sampling/createMessage` can use the already loaded OmniCortex, after the MCP interaction host has reviewed its input. The native adapter does not load another model, invoke a teacher API, call ordinary chat, append a persona, or replay old chat text. Output is returned to the host for a second review before the server receives it.

The adapter accepts reviewed text, inline image, and inline audio blocks, including block arrays. Its output is text. Tools, extra context, model preferences, and system prompts are rejected at the authenticated native boundary, never silently obeyed. The MCP interaction host may display advisory model preferences and an explicitly omitted server system prompt, but neither becomes hidden native instructions. Additional server context, sampling tools, video input, and generated media output are not supported by this adapter and must not be advertised.

Every request preserves the exact reviewed JSON, roles, block positions, MIME spelling, and binary source bytes in its approval hash. Native output receipts are bound to the same brain/request/hash. Each media receipt additionally binds its position, type, MIME, raw-source SHA-256, source-byte count, actual decoded-window count, complete traversal, and actual native sensory processing. The bridge rejects a missing, substituted, or incomplete receipt.

The protocol allows client-controlled model selection and reviewed sampling. These protocol details are distinct from this application's native execution and permission policy. [Official MCP sampling specification](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling).

## Ownership and deadlock avoidance

- `EngineSupervisor.sampleNativeMcp()` validates the exact approved JSON/hash and uses metadata-only enqueue/status/cancel/consume controls.
- A request nested inside a live MCP tool call carries the trusted controller turn ID. The same existing neural writer services it at a cooperative boundary, including its external-result wait. It never queues another chat behind the turn awaiting the tool.
- Idle requests use serial `mcp_sample` execution and require an already loaded native identity. The stdin reader never performs inference or learning.
- Requests retain their exact status until main consumes the terminal receipt. A request ID reused with different reviewed bytes is rejected.
- Stop, Steer, Unload, server/session cancellation, or task closure stops the private request. Interrupted partial output is not delivered as a successful sample.
- The tool's configured external wait allowance excludes time spent performing the nested native computation; no synthetic tool observation is manufactured when the actual result is missing.

## Private execution

The adapter shares authoritative packed cortical, VSA, liquid, and LIF owners. It creates no floating-point learned master and does not clone a full model.

Only small activity buffers are temporarily replaced with private tensors. Native generation owns its cache and RNG. A shallow neural-memory view shares authoritative vectors and synapses while owning disposable recall frontiers, graph readers, and inspection metadata. Recall is called with activity recording disabled; each consumed private frontier is closed immediately. Observer bindings, mixed module modes (including modality regions), original activity tensor identities, and global RNG are restored on success, interruption, or failure. The existing outer generation cache, tokens, pager cancellation scope, and parameter journal remain owned by the original turn.

A native read-only parameter boundary rejects learned writes before an enclosing turn's diagnostic journal sees them. No episode, replay, counter, event, source blob, or chat row is created by sampling. Any later learning of an approved real tool result belongs to the normal host outcome path, not this private candidate.

`maxTokens` is preserved, not reduced to a test-sized cap. The complete requested token/result allocation must fit actual resource admission. Oversized token input is rejected without truncation. Shared VSA cue reduction now uses bounded blocks (up to 32 rows or 64 KiB, with a minimum full row), retaining every label and position. The sampler admits that reduction separately from labels, weighted-row/contact decoding, model activity, and media decoding; the estimate is not a total-RSS guarantee. The short reduction path remains unchanged; long compensated FP32 reduction has tested numerical parity, not a bit-for-bit claim. Stop sequences are matched across complete UTF-8 emissions with a bounded suffix, not repeated whole-prefix decoding. Results distinguish actual learned EOS, token budget, and a matched stop sequence.

## Same-brain image/audio input

Supported image MIME types are `image/png`, `image/jpeg`, `image/webp`, and `image/gif`. Supported audio types are `audio/wav`, `audio/x-wav`, `audio/wave`, `audio/flac`, `audio/x-flac`, `audio/ogg`, `audio/mpeg`, `audio/mp3`, `audio/aac`, `audio/mp4`, and `audio/x-m4a`. MIME selection is case-insensitive, while receipts preserve the approved spelling. Only canonical inline base64 is accepted: a server cannot grant a local pathname, URL, playlist, SVG, or arbitrary decoder input.

Sources decode in bounded copy windows into admitted RAM first. Actual authorized spill is used when required, including a private seekable source when the audio codec requires one. The private filename is generated locally, created exclusively without following symlinks, charged to the existing storage quota, and removed on completion, rejection, or cancellation. Reopened codec paths must retain their original regular-file identity; cleanup rejects a replacement and never deletes that replacement. This is not a sandbox against a hostile process already running with the same operating-system account.

Images reserve the declared canvas and native-frame scratch before pixel conversion, visit every admitted animation frame, and resize into the brain's existing sensory geometry. Audio visits every decoded source block, downmixes channels, uses the existing phase-continuous resampler, and processes every native-sized window including the padded tail (padding is excluded from actual sample counts). Multiple/chained Ogg streams and unsupported multiple audio tracks fail explicitly rather than silently dropping data.

Pillow and SoundFile handle their supported containers; the existing codec gateway supplies the audio fallback. That fallback forces the declared demuxer, permits only `file`/`pipe` protocols, disables MOV external data references and absolute paths, and has no shell. Its bounded reader queue is I/O only, never a second neural writer. Decoder cancellation and resource pressure terminate the private child and drain its reader. The codec allocation allowance follows admitted resources and is not represented as an OS-enforced whole-process RSS limit. [Official FFmpeg format options](https://ffmpeg.org/ffmpeg-formats.html).

Each actual frame/window calls the **same loaded brain's** `ModalityHub.perception_embedding()` and its native sensory projection, private liquid state, read-only associative recall, and LIF routing (`learn=False`). Resulting activity conditions private language generation directly. No caption, source-filename description, fabricated instruction, ordinary ingestion call, episode, or learning update substitutes for sensory processing. Media-only requests use language-boundary role tokens, not a hidden “describe this” prompt. Disabled modality regions reject the input before decoding. Only successful whole-input traversal plus actual sensory encoding produces a media receipt.

The host's pre-generation review presents bounded metadata (type, MIME, bytes, hash), not automatic untrusted pixel/audio decoding in Electron. It preserves exact media during editable text review. Native output then receives the separate post-generation review before delivery to the MCP server.

## Verification scope

`engine/tests/test_mcp_sampling.py` uses one small actual native random-weight fixture with an actual learned graph/vector experience. This is a mechanics proof, not conversational intelligence qualification. It covers unchanged packed state and vectors/graph, counters, pending learning/events, messages, RNG, mixed modes, and original activity owners; no old-history prompt injection; UTF-8 stop sequences; private cache cleanup; active Stop/Steer/Unload; a real interleaved main cache; and nested sampling serviced on the waiting writer rather than the reader thread.

`engine/tests/test_mcp_sampling_media.py` separates decoder/transport fixtures from actual neural proofs. Decoder fixtures cover PNG/JPEG/WebP/GIF, WAV 8/16/24/32-bit input and resampling/tails, FLAC/Ogg, actual local MP3/AAC/M4A decoding, corruption/admission refusal, source cleanup, blocked-reader cancellation, and backpressure errors. The codec fixture uses an already installed local codec binary; it does **not** prove production download/signature setup or packaging on every OS. A separate fixture verifies codec owner/cancellation correlation without claiming native inference.

The actual native media fixture uses the real packed OmniCortex and its real sensory encoders: two GIF frames plus all three windows of a 137-sample WAV. It checks packed/vector/graph bytes, pending events and learning, messages, counters, original activity identities, mixed modes, RNG, and pager cleanup. Media-only, cancellation, disabled-region rejection, and shared bounded VSA scratch admission are separately covered. The source-path replacement fixture verifies rejection without deleting foreign replacement bytes.

Recorded bounded results (individual affected runs, not one whole-project suite):

- 14 basic native sampling cases passed before the media extension; existing cooperative generation, inbox, and transport regression slices passed.
- After media changes, the actual native read-only sample, nested waiting-writer sample, and original outer-cache/pager-scope sample passed again in a four-case run (17.224 s). That run's new scratch-admission case initially failed due solely to a misplaced test assertion; after correction, that case passed individually (2.629 s).
- Three actual native sensory cases passed (6.795 s): image/audio, media-only role input, and cancellation/disabled-region handling.
- Twelve decoder cases, codec-correlation fixture, and actual authorized-drive cleanup fixture passed (14 cases, 1.039 s) before the final path-identity hardening. Seven affected decoder/cleanup cases then passed (1.173 s), including the new replacement-file rejection, RAM/spill cleanup, actual local MP3/AAC/M4A decoding, cancellation, and backpressure-error handling.
- The 10 TypeScript native bridge tests passed (689 ms), including exact binary-source receipts and substitution rejection. Node TypeScript checking passed.
- Python compilation of the affected native/worker modules and the repository whitespace check passed after the final decoder change.

Transport and wait-budget regressions are in `test_worker_transport.py` and `test_chat_tool_observation.py`; bridge tests are in `tests/nativeMcpSampling.test.ts`. These scoped fixtures do not establish broad learned intelligence, production throughput, all-platform runtime provisioning, remote MCP service compatibility, full user-corpus training, or the overall project goal's completion. Sampling tools, extra server context, video input, and media output remain explicit unsupported protocol modes, not hidden fallbacks.
