# Separate assistant and pre-action reviewer

R236 and R094 are implemented additively in the existing Electron application.
They do not replace OmniCortex, import foundation weights, teach it implicitly,
or supply behavioral prompts to its native decoder. Both channels default off.

## Implementation

- `externalAdviceClient.ts`: five explicit text-only provider adapters, bounded
  streaming and terminal validation, cancellation, a killable 120-second default
  deadline, redirect/tool/fallback refusal. OpenAI uses Responses with `store:false`;
  Anthropic uses Messages; Gemini uses streamGenerateContent; OpenRouter and an
  explicit loopback local server use chat-completion streams. No real provider,
  account, key or local model was used for these checks.
- `externalAdviceService.ts`: separate channel configuration, explicit request
  preview, settings/context/device revision fencing, partial-history custody,
  no replay on restart, and known-secret redaction before transmission/history,
  including another provider's secret split across output deltas. Device-private
  secure-store references do not enter brain storage. A parsed `recordReview`
  sink also retains timeout/unavailable reviews without a provider result.
- `externalAdviceAuthority.ts`: installation-only, per-brain consent. Import,
  merge and copy cannot gain it from portable enabled preferences. Settings
  changes revoke first and activate only after a successful explicit save;
  recovery retires prior consent and awaits cancellation before promotion.
- `externalAdviceContext.ts`: latest actual human task and recent recorded
  replies/action status from a 32-entry ledger window, under the selected budget.
  Omitted/partial history is disclosed; MCP-authored human-role inputs are not
  relabelled as human instructions. Tool arguments are not appended as generic
  activity. This is external consultation context, not native memory retrieval.
- `actionReviewer.ts` and ToolExecutor: at least one second plus completed review,
  strict whole JSON and exact observed-quote evidence, bounded timeout/Stop,
  frozen arguments, original/resolved plugin operation, current owner/config/
  permission and trusted chat-action identity binding. Flagged or failed review
  uses the existing one-use Ask approval. An advisory allow cannot grant tools.
  Historical committed effects are delivered without another dispatch/review.
- Main/preload/IPC mount the separate panel and review details in the existing
  workspace/action cards and manual Tools console. Decline consumes one exact
  approval without dispatch/learning; a different instruction becomes an ordinary
  composer draft without overwriting current text. Manual Tools can discard its
  exact approval. Optional controls do not initialize the native brain or a second
  local model. Off review avoids creating an advisory database on every tool call.

Complete portable settings, histories, parsed receipts, ancestry and same-ID
recovery are covered separately in [advice custody](external-advice-portability.md).
Keys, host grants, live external requests and approval authority remain separate.

## Source checks

Only distinct new affected cases ran; previously closed suites were not repeated.

| Slice | Evidence |
| --- | --- |
| Five provider stream adapters and service | 13 fake-fetch/temp-SQLite cases; timeout and cross-delta credential echo included |
| Parsed standalone review custody | 1 new case, 41 ms |
| Review/ToolExecutor authority | 25 distinct fake-service cases across initial and affected-only runs; no native brain |
| Device grants and recorded context | 3 distinct cases; one initial over-specific test budget was corrected; final task-role and MCP-origin boundaries checked |
| Exact chat decline | 1 constructor-free controller case, 6 ms; no dispatch or learning |
| Portable history/recovery | 7 distinct SQLite cases; same-owner review-ID bug fixed with only its affected case rerun |
| User curriculum | 1 new pack-16 declaration case, 4 ms; same eight ordinary documentation records, no execution/persona/reward loop |
| Response resources/recovery | 3 new cases,236ms:200 deltas with bounded checkpoint admissions; pre-dispatch disk refusal/zero provider calls; orphan retirement without replay. One directly affected earlier suspend case reran25ms, other5 skipped |
| Live versus historical UI state | 2 new pure-presentation cases,3ms: saved pending prefixes never imply a live provider; stale history cannot relock a completed view; received text and saved prefix remain distinct |

Both repository TypeScript configurations passed after the coordinated source
integration. Subsequent source changes are noted in the continuation rather than
treated as proof of a packaged app. No GitHub run, package, release, long training,
model generation or paid request is part of this scoped evidence.

## Native component UI check, October 13

Actual Electron window, `tests/e2e/external-advice-main.cjs`, isolated disposable
profile. This fixture mounts the actual added components with an explicitly
synthetic API: no native brain, real key, network request, real tool or grant.
It is not a replacement application. The 380-pixel narrow content surface is
inside a 1100-pixel native window, not a physical 380-pixel app-window test.

| Step | Observed control/health | Accepted capture |
| --- | --- | --- |
| 1 | Separate assistant surface, disabled initial send, settings disclosure; readable light appearance | `01-assistant-light.jpg` (before explicit disabled-opacity correction) |
| 2 | Explicit device checkbox/save and typed prompt registered; exact request and 42 omitted records appeared before Send | AX evidence; capture02 shows lower settings, not full preview |
| 3 | One synthetic Send produced a separate saved assistant reply; input cleared and Send disabled | `04-saved-reply-light.jpg` |
| 4 | Dark narrow panel retained the reply, wrapped text and visible channel selection | `05-assistant-dark-narrow.jpg` |
| 5 | Narrow reviewer settings and lower content inspected; cramped checkbox label given an explicit wrapping span | `07-reviewer-label-wrap-correction.jpg`;06 is pre-correction |
| 6 | Tab from settings disclosure reached its actual enable checkbox; native focus identified by AX | AX evidence only |
| 7 | Decline-and-draft accepted typed user text and showed an inert synthetic denial/draft receipt | `08-decline-and-draft-dark-narrow.jpg` |
| 8 | Isolated UI window closed and its Electron process exited0 | supervised process completion |

Captures are in `test-results/external-advice-ui-2026-10-13/`. Accepted saved files
were reopened and inspected. Capture03 is an incomplete scroll view and is not
accepted as saved-reply proof. No provider response or executed tool is implied.
The screenshot-based audit preserved the existing theme tokens and found/fixed
the narrow label and disabled-button presentation; React review checked scoped
async ownership, subscription cleanup, manual selection and escaped text output.

### Affected resource/recovery follow-up

Every advice workspace mutation now requires the injected production disk guard
(real20GiB reserve) and a conservative SQLite/journal amplification estimate.
Missing admission fails closed. The initial per-delta FULL-sync response rewrite
was replaced by64KiB/500ms checkpoints; events still carry received redacted text
with explicit saved-prefix counts. Terminal, cancel and suspend flush. Resource
refusal aborts the request, preserves the last admitted prefix and never writes
past the guard. This is a bounded source comparison, not a disk-life or whole-app
throughput benchmark.

History retires orphaned prior-session pending rows without replay. Current
`activeRequestIds`/event flags—not historical queued/streaming state—control Send
and Stop. The panel labels an inactive prefix as saved history rather than a
provider still loading. Source initialization fencing prevents a concurrent
history query from retiring this service's newly starting request.

One affected native component UI-only check displayed the synthetic saved prefix,
no live provider/Stop control, and accepted new typed input with Preview enabled.
It did not send a second request. Capture09 (`09-interrupted-prefix-no-live-lock.jpg`)
was saved/reopened/inspected; its temporary Electron process exited0. Earlier UI
flows were not repeated. Final semantic checks cover this source revision.

## Still open

Real provider/account/local-server interoperability and model judgment; actual
loaded-brain-to-tool review/approve/Stop sequencing; complete app overlay/hover,
zoom, all-theme/accessibility and packaged-platform checks. Protocol success
does not establish detection of hidden intentions, universal safety, or native
learned tool/memory capability. Broad trained capability remains deferred.

Protocol references: [OpenAI streaming](https://developers.openai.com/api/docs/guides/streaming-responses),
[Anthropic Messages streaming](https://docs.anthropic.com/en/api/messages-streaming),
[Gemini streaming content](https://ai.google.dev/api/generate-content),
[OpenRouter streaming](https://openrouter.ai/docs/api/reference/streaming).
