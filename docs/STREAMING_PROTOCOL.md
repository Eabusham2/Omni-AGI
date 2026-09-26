# Neural chat streaming contract

Omni's worker API is stable protocol version **1** (the product's `1.0`
contract). Its wire envelope uses JSON-RPC **2.0**, because ordered progress
and cancellation events rely on standard notifications. “Protocol 1.0” in the
product plan refers to the Omni API version, not a JSON-RPC 1.0 envelope.

The desktop sends a unique `streamId` in every `chat` JSON-RPC request. While
that request is active, the worker may emit JSON-RPC notifications using the
existing `event` method:

```json
{
  "jsonrpc": "2.0",
  "method": "event",
  "params": {
    "type": "chat-token",
    "brainId": "brain-id",
    "streamId": "turn-id",
    "sequence": 0,
    "data": { "delta": "A bounded UTF-8 text delta" }
  }
}
```

```json
{
  "jsonrpc": "2.0",
  "method": "event",
  "params": {
    "type": "chat-action",
    "brainId": "brain-id",
    "streamId": "turn-id",
    "sequence": 1,
    "actionId": "worker-local-action-id",
    "data": {
      "action": {
        "kind": "imagine",
        "toolId": "modality.imagine",
        "action": "generate",
        "arguments": {
          "modality": "image",
          "conceptIds": ["active-assembly-id"]
        },
        "confidence": 0.84
      }
    }
  }
}
```

```json
{
  "jsonrpc": "2.0",
  "method": "event",
  "params": {
    "type": "modality-preview",
    "brainId": "brain-id",
    "jobId": "runtime-job-id",
    "streamId": "turn-id",
    "sequence": 2,
    "actionId": "worker-local-action-id",
    "progress": 0.35,
    "message": "Decoding the current latent",
    "data": {
      "preview": {
        "schemaVersion": 1,
        "revision": 0,
        "modality": "image",
        "stage": "diffusion-vq-decode",
        "mimeType": "image/png",
        "mediaUrl": "omni-media://artifact/<lease>/<sha256>",
        "completedUnits": 1,
        "totalUnits": 4,
        "producer": "same-brain-decoder",
        "payloadSha256": "sha256-of-exact-preview-bytes",
        "actualDecoderOutput": true,
        "spatialResolutionReduced": false,
        "cadence": "hardware-aware-bounded-synchronous"
      }
    }
  }
}
```

After the last visible token, the worker emits one provisional phase event
immediately before synchronous post-response learning and save work:

```json
{
  "jsonrpc": "2.0",
  "method": "event",
  "params": {
    "type": "chat-phase",
    "brainId": "brain-id",
    "streamId": "turn-id",
    "sequence": 3,
    "data": {
      "phase": "reply-complete-learning",
      "replyComplete": true,
      "turnCommitted": false,
      "learning": true,
      "saving": true
    }
  }
}
```

This event does not complete the turn. The renderer keeps the generated text
and measured token count visible, keeps ordinary composer input queue-only,
and waits for terminal `chat-state: complete` before allowing another parallel
send. No progress percentage is inferred for learning or persistence.

After a visible external action completes, the trusted controller may emit a
second `chat-phase` with `phase: "action-result-learning"` and
`turnCommitted: true`. This means the original human/reply pair is already
committed and the completed action result is entering typed neural ingestion.
The serialized result is never submitted as another human chat message, never
adds a hidden reply, and cannot recursively propose more actions. Any later
natural-language continuation requires a separate explicit protocol.

`sequence` is a unique, monotonically increasing non-negative integer within
one stream. The supervisor drops missing, duplicate, and out-of-order
sequences, as well as notifications with another `streamId`. Action values are
accepted only from `chat-action` or the final typed `actions` result. Response
prose, tags, slash commands, and hidden prompts are never parsed into actions.

Reversible app navigation uses the same typed channel. For example,
`{"kind":"tool","toolId":"studio.ui","action":"open-creativity","arguments":{}}`
opens the local Imagination workspace only after its trusted execution reaches
`complete`. The renderer does not inspect chat prose for this route. Because
the route changes only local UI state, it is always available at `auto` and
does not inherit file, process, network, browser, or source permissions. Its
proposed, running, and completed states remain visible as an action card and
the completed invocation is written to the operational tool audit.

The worker commits every preview to a content-addressed brain-local cache and
passes its absolute path only to the trusted main process. The main process
validates the brain root, MIME signature, file identity, and SHA-256 before it
issues a short-lived unguessable `omni-media://` lease. Filesystem paths and
large base64 values never cross preload/renderer IPC. Sub-64 KiB data URLs are
retained only for tiny compatibility fixtures and browser-only demos.

The main process bounds token deltas, labels, paths, and tiny preview data URLs. A
preview is display-only; only the terminal tool result is integrated into
neural experience. Image revisions are actual diffusion-step VQ decoder
outputs, audio revisions contain an increasing prefix of the actual neural
codec waveform, and video revisions contain an increasing timeline of actual
temporally decoded frames. The worker's synchronous bounded callback provides
backpressure rather than accumulating preview tensors. Hardware tier changes
only the number of intermediate revisions; it never lowers decoder work,
spatial resolution, duration, frame count, or final quality. Cancelling a turn
aborts its neural request, queued tools, and active modality job while retaining
the latest visibly marked partial revision. On completion, the final artifact
replaces the display-only revision through the existing job/action channel.
The renderer displays the worker's zero-based `revision` as `rN` unchanged;
`payloadSha256` binds that revision to the exact encoded preview bytes, so it
does not synthesize an extra ordinal revision after hot reload or completion.

Completed artifacts are content-addressed under the brain's artifact directory
and recorded in a checksum-bound artifact index. Gallery requests page that
index and lease only the visible page, so restart, export/import, and histories
larger than 512 artifacts remain usable without returning already-revoked URLs.

## Paged conversation history

The neural engine is authoritative for committed human/brain messages and
measured traces in its append-only hash-chained SQLite ledger. Electron keeps a
separate stable-ID presentation ledger for those exact receipts plus host action
lifecycle; it never invents or renumbers neural rows. Both `brain.json` files
store only counts, head cursor/hash, and current attention epoch—not monolithic
message or trace arrays. Legacy arrays are backfilled once in stable timestamp
order.

`chat.listPage` returns at most 200 message/action rows before an exclusive
sequence cursor. The renderer requests 120 rows, renders at most 120 dynamic
height rows, preserves a surviving row's viewport offset when prepending, and
pins current pending/failed activity while an older page is open. Full history
remains in the ledgers and is searchable through `brain.history`; windowing is
presentation state, never deletion or truncation.

## Typed mid-turn steering

The composer remains available while a response is active. Ordinary Enter
queues that text until the terminal turn event, so it cannot overwrite or run
in parallel with the active turn. Ctrl/Cmd+Enter is the explicit typed
replacement operation: the renderer cancels the exact active `turnId`,
allocates a new turn, and sends the person's new text unchanged with
correlation metadata:

```json
{
  "kind": "steer",
  "replacesTurnId": "cancelled-turn-id",
  "source": "human",
  "createdAt": "2026-08-10T00:00:00.000Z"
}
```

This metadata is transport and presentation state only. It is validated at the
preload/main boundary, appears on the replacement turn's `chat-state` event and
terminal `ChatResult`, and is never concatenated into model input, a hidden
prompt, or long-term neural memory. Late output from the replaced turn is
discarded by turn/generation correlation. A bounded, expiring cancellation
tombstone handles the race where cancellation reaches the main process before
the delayed replacement's predecessor has registered.

## Voice delivery

Voice uses the same ordered token stream and exact turn cancellation. `live`
delivery speaks bounded sentence/character chunks while generation continues;
only one native speech utterance is active and at most three waiting chunks are
retained by default. If generation outruns speech, obsolete queued audio is
replaced and the skipped-character count is shown without dropping generated
text. `buffered` delivery waits for the terminal reply and deliberately ignores
microphone-triggered barge-in; the visible Stop control still cancels the exact
turn. Slow, normal, and fast pace values affect speech presentation only.

Platform Web Speech recognition and platform speech synthesis are the default
adapters and are labelled as such. Neural listening is an independent,
capability-gated sensory stream: raw microphone audio forms assemblies and
STDP changes in the same persistent brain, while Web Speech still supplies the
literal transcript at the language boundary. Generic audio generation is not
intelligible TTS, so neural voice remains unavailable unless a verified speech
decoder explicitly provides it. Platform adapters are never relabelled or
silently substituted as neural.
