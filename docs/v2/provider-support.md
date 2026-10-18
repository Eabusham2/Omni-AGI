# Optional OpenRouter and Ollama-local provider support

Internal source evidence, October 15, 2026. This is optional external assistance
and teaching, not another native brain backend. No real key, provider request,
Ollama service/model, neural worker, training, remote CI or package ran for this
slice. Root owns app/IPC/preload composition and actual synthetic UI evidence.

## Implementation

`src/main/apiTeacherProviders.ts` describes existing HTTP routes and request
shapes without importing an AI SDK or starting anything. The shared teacher
provider union now includes `openrouter` and `ollama`. The existing three
teacher-provider descriptions were initially preserved when moved into the
helper; the October 18 Gemini correction below supersedes that provider's
earlier text shape. Durable teacher orchestration, consent, request
hashes, uncertain dispatch and learning receipts are the teacher owner's work;
the helper/discovery tests do not prove that a brain learned.

### October 18: Gemini text-response correction

Current [Gemini Interactions REST reference](https://ai.google.dev/api/interactions-api)
and its [official OpenAPI schema](https://ai.google.dev/static/api/interactions.openapi.json)
were checked directly. The existing Gemini text helper incorrectly inspected
`output` and omitted the requested output-token budget. The adapter now sends
`generation_config.max_output_tokens` with the selected budget and retains
`store: false`.

Only a `completed` response with typed `steps` is accepted. Text is extracted
in order from `model_output.content` text blocks; actual `thought` summaries
are not substituted for final lesson output. Tool/user-input steps, non-text
blocks, malformed or errored output and partial/action-required status fail
explicitly, even if an `output_text` lookalike is present. No legacy output
fallback or automatic tool execution was added. The separate typed-media
parser and other four providers' text branches are unchanged.

Six new pure cases in `tests/apiTeacherGeminiTextShape.test.ts` passed in 4 ms:
documented request budget, ordered text, thought exclusion, incomplete/error
status, unsupported instructions/wire shapes, and other-provider callback
preservation. One obsolete Gemini request expectation in the older provider
fixture was updated, but that old suite was not rerun. This is source-shape
evidence only: no key/account, provider generation, model, native learning,
compiler, CI or package ran for this correction.

The separate assistant and pre-action reviewer both accept explicit Ollama-local
configuration. Their existing text-stream client and SQLite custody remain;
no external model is silently substituted for OmniCortex. Portable history and
settings contain no credential or current device grant. Ollama needs no API
key, and an unrelated credential is refused rather than forwarded to it.

### OpenRouter

- Chat/teacher dispatch uses fixed
  `https://openrouter.ai/api/v1/chat/completions`, the exact selected model,
  `max_completion_tokens`, and provider fallback disabled. Existing advisory
  SSE remains the same transport. The official [chat API](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion)
  documents this HTTP contract; it is not evidence of an actual paid call.
- Model discovery uses the user's key with the fixed **account-filtered**
  `https://openrouter.ai/api/v1/models/user`, honoring the endpoint intended for
  provider/privacy/guardrail preferences. It never silently retries the global
  catalog. The official [account-filtered models API](https://openrouter.ai/docs/api/api-reference/models/list-models-filtered-by-user-provider-preferences-privacy-settings-and-guardrails)
  documents offset/limit pagination. Source uses pages of100, fixed-origin
  offsets, repeated-page refusal and existing response-byte/deadline bounds;
  provider-supplied pagination URLs cannot redirect a key elsewhere.
- Catalog results are untrusted metadata, not compatibility or safety proof.
  The manual model field remains available and listing does not choose a model.

### Ollama local

- Teacher dispatch defaults to
  `http://127.0.0.1:11434/v1/chat/completions`, using the existing compatible
  HTTP/SSE shapes. Advice additionally accepts an explicit loopback HTTP(S)
  base or `/v1` endpoint. Non-loopback destinations, embedded credentials,
  query/fragment overrides and arbitrary API paths are refused. The official
  [Ollama compatibility documentation](https://docs.ollama.com/api/openai-compatibility)
  describes local chat and `max_tokens`; no SDK is required by this implementation.
- Discovery reads the selected local server's `/api/tags`, as documented in
  [Ollama's installed-model inventory API](https://docs.ollama.com/api/tags).
  There is no key lookup for this local discovery, automatic server startup,
  model pull, or native model Load. An explicit later assistance/teacher request
  can naturally cause the user-managed server to load the chosen installed model.
- Known `:cloud`/`-cloud` selections are refused with a separate-cloud-consent
  explanation, before provider dispatch. The teacher owner also validates this
  before durable job creation. Inventory entries advertising remote execution
  are excluded from local selection, with every excluded ID/reason inspectable
  through paged controls. They are not silently omitted.
- A loopback URL or inventory response is **not proof of fully offline model
  execution**. Custom/obscured server aliases and server-side forwarding cannot
  be certified from this metadata. Users must run a trusted local setup; no
  automatic cloud consent is inferred. Ollama cloud is not added as a hidden
  fallback. The generic OpenAI-compatible local option retains manual IDs rather
  than guessing its catalog route.

## Advice-owned discovery and stale authority

`ExternalAdviceApi.discoverModels(brainId,{channel,requestId,
expectedSettingsRevision,expectedCredentialRevision})` reads provider/endpoint
from the exact saved channel. Result includes the same owner/channel/request,
settings and credential revisions plus the exact local endpoint. The service's
credential shim reads **advice-owned opaque device keys**, never teacher keys.
No task/context or conversation text is included in model-list requests.

Requests are registered before asynchronous config/key lookup. Exact queued
Stop therefore prevents late dispatch. Config, key, channel/device authority,
Unload/recovery and service retirement cancel pending discovery; post-await
fences reject even a fake transport that ignores Abort. Cross-channel or old
request IDs cannot cancel a replacement owner. Model listing creates no
durable execution grant or inherited external handle.

`ExternalAdviceModelPicker` performs discovery only after a human click, not on
mount from imported settings. It validates complete owner/provider/endpoint and
revision identity before displaying results. Manual IDs remain. Both eligible
and excluded lists use bounded visible pages without a total-record ceiling.
Unsaved draft provider/endpoint changes also change the actual React mount key,
retiring pending old scopes immediately; saved-config-only fencing would not
have covered that UI edge. Typed result validation and this actual mount are
source evidence, not a new browser-interaction result from this lane.

The React checklist was applied to primitive lifecycle dependencies, independent
request scopes, stale responses, plain-text rendering, labels, controls and
bounded DOM lists. It did not introduce a component framework or replace the
existing Electron layout.

## Focused evidence and limits

**11 distinct new cases pass**, counted once:

- `tests/apiProviderSupport.test.ts`:5 fake/pure cases covering HTTP descriptions,
  existing-provider body preservation, exact response extraction, no-key local
  inventory, inspectable cloud exclusions, account-filtered pagination,
  local streamed text, cloud/key/redirect refusal, and keyless
  per-channel history under current device authority.
  The history fixture originally timed out because its fake client returned
  text without emitting matching deltas. Source correctly did not report a
  matching completed stream. The stub was corrected and only that affected
  case reran:1/1,26ms. Later2 affected HTTP cases passed17ms after accepting
  protocol-null/empty tool slots while still refusing actual tool requests.
  The original timeout is not retroactively a production pass.
- `tests/adviceModelDiscovery.test.ts`:3 cases passed194ms. They prove separate
  advice keys, exact saved/current revisions, no context/model call, early
  exact Stop, wrong-channel refusal, custom loopback binding, late config/key
  rejection even when injected HTTP ignores Abort, and manual generic-local
  behavior. Only owned temporary SQLite/files and fake fetch are involved.
- `tests/apiProviderUi.test.tsx`:3 static/pure cases passed118ms. Provider choices,
  keyless local metadata controls, paged inspectable exclusions, exact
  owner/channel/revision/credential/endpoint result refusal and explicit-click
  controls are covered. Server-side rendering does not execute effects or
  prove real UI interaction/IPC/provider delivery.
- Scoped owned TypeScript checks passed after correcting fake secret-store
  `remove()` return types. No broad old suites were rerun. Parent coordinated
  compilation and live synthetic UI observations are recorded separately.

Real account authentication, provider availability/billing, local-server
interoperability, actual model outputs, teacher neural updates and learned
quality remain unverified here. The implementation never reports catalog
listing, streamed fake text or a completed metadata request as those proofs.
