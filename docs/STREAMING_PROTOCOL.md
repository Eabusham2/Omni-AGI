# Neural chat streaming contract

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
        "revision": 0,
        "mimeType": "image/png",
        "dataUrl": "data:image/png;base64,...",
        "artifactPath": "artifacts/preview-0.png"
      }
    }
  }
}
```

`sequence` is a unique, monotonically increasing non-negative integer within
one stream. The supervisor drops missing, duplicate, and out-of-order
sequences, as well as notifications with another `streamId`. Action values are
accepted only from `chat-action` or the final typed `actions` result. Response
prose, tags, slash commands, and hidden prompts are never parsed into actions.

The main process bounds token deltas, labels, paths, and preview data URLs. A
preview is display-only; only the terminal tool result is integrated into
neural experience. Cancelling a turn aborts its neural request, queued tools,
and active modality job while retaining visibly marked partial output.
