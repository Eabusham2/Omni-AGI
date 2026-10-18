# API teacher model discovery

Omni AGI Studio can read the model catalogue exposed to a user-supplied API
credential for OpenAI, Anthropic, or Gemini. Discovery is a settings aid for an
external teacher. It does not replace OmniCortex, select a model, start a
teacher job, call generation, prove endpoint compatibility, or change learned
state.

## Provider contracts

- OpenAI uses one fixed `GET https://api.openai.com/v1/models` request and the
  returned model IDs/basic metadata described by the
  [official List models reference](https://developers.openai.com/api/reference/resources/models/methods/list).
- Anthropic uses `GET https://api.anthropic.com/v1/models` with
  `anthropic-version: 2023-06-01`, page size 1000, and the documented
  `after_id` / `last_id` / `has_more` cursor sequence from the
  [official List Models reference](https://platform.claude.com/docs/en/api/models/list).
- Gemini uses `GET https://generativelanguage.googleapis.com/v1beta/models`,
  page size 1000, and the opaque `pageToken` / `nextPageToken` sequence from
  the [official Models API reference](https://ai.google.dev/api/models).
  Authentication uses the `x-goog-api-key` header described by the
  [official Gemini API reference](https://ai.google.dev/api).

Provider-returned order is preserved. Gemini exposes its resource name and the
documented `baseModelId`; the latter is the suggested manual model value. A
listed model can still be unavailable for the teacher endpoint, account,
region, billing state, input, or requested operation. Every result therefore
states `generationCompatibilityVerified: false`.

## Credential and request boundary

Credentials remain in the existing installation-private `SecureSecretStore`.
Discovery makes no network request when the selected provider is unconfigured.
Requests use fixed HTTPS origins, `redirect: error`, bounded two-MiB pages and
an explicit eight-MiB cumulative resource refusal. Exceeding that budget fails
the discovery; it never truncates a catalogue and reports it complete.

The renderer supplies a bounded request ID and provider. Repeating the same
active ID for that provider shares one in-flight request. Cancellation is bound
to the same pair. Pagination rejects repeated cursors. Credential replacement
or removal aborts matching work; main rereads and verifies the credential
revision and digest immediately before returning a result, so stale responses
are discarded. Results never contain a credential or authorization header,
and provider/fetch errors redact the known secret.

The manual model field remains authoritative. Discovery returns candidates for
review but never overwrites that field or silently launches generation.

## Evidence boundary

Six focused cases use fake HTTP responses and an in-memory fake secret reader.
They cover no-network unconfigured behavior, the fixed OpenAI request,
Anthropic and Gemini pagination, metadata projection, request-ID cancellation,
redirect refusal, provider-error redaction, and credential-change retirement.
No real key was read or written and no provider, model, generation, training,
account, billing, region, or compatibility check ran.
