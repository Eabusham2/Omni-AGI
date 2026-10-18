# Structured questions

Omni can ask a single-choice question, a checkbox question with a required minimum, or a written question. Recommended options appear first, followed by their ranking. Choice questions allow an Other response by default; callers can disable it when a minimum set of listed choices is required.

Questions and answers belong to the instance and survive closing the application. Answering twice with the same selection is safe; a different second answer cannot overwrite the recorded response. Cancelling a question records that choice without inventing an answer.

## Timeout behavior

Automatic assumptions are off by default. When enabled, the default wait is 20 minutes; the setting stores a positive number of seconds so the interface can offer minutes and seconds. Changing the setting updates pending questions and starts the new wait interval from the change. Disabling it removes pending deadlines.

At expiry, a choice question uses its explicit suggested answer or enough highest-ranked options to meet its selection requirements. A written question needs a suggested written answer; without one it remains waiting. The saved result states that it was assumed by timeout and records its deadline and any overdue duration, including time while the application was closed.

These answers only choose how to proceed with a task. Every result contains `authority: "task-choice-only"` and `permissionsGranted: false`. Neither a user answer nor a timeout changes tool permissions, provides an approval token, or authorizes a destructive operation. Those actions continue through their actual permission and approval controls.

## Host API

`src/shared/userQuestions.ts` contains the renderer-safe types and `UserQuestionsApi`. `src/main/userQuestions.ts` exports `UserQuestionService`.

```ts
const questions = new UserQuestionService({
  brainDirectory: (brainId) => repository.brainDirectory(brainId),
  onError: (error, brainId) => reportQuestionError(brainId, error)
});

const question = await questions.request({
  brainId,
  requestId: actionId,
  prompt: "Which sections should I update?",
  mode: "multiple",
  minSelections: 2,
  allowOther: false,
  options: [
    { id: "intro", label: "Introduction", recommended: true, rank: 1 },
    { id: "examples", label: "Examples", recommended: true, rank: 2 },
    { id: "appendix", label: "Appendix", rank: 3 }
  ],
  source: { kind: "brain", turnId, actionId }
});
```

| Method | Result |
| --- | --- |
| `request(request)` | Creates a durable request. Reusing its request ID with identical content returns the saved record; changed content is rejected. |
| `list(brainId, {status?})` | Lists saved requests in creation order. Reconciles pending deadlines first. |
| `get(brainId, questionId)` | Reads one owned request. |
| `respond({brainId, questionId, response})` | Validates and commits selected option IDs or written text. |
| `cancel(brainId, questionId, reason?)` | Resolves a pending request as cancelled. |
| `preferences(brainId)` | Reads the instance's timeout settings. |
| `setPreferences({brainId, timeoutEnabled, timeoutSeconds})` | Saves settings and reconciles existing pending questions. This is user-controlled configuration, not a native tool argument. |
| `subscribe(listener)` | Subscribes to durable creation, update, resolution, cancellation, delivery, preference, and error events; returns unsubscribe. The preload API calls this `onEvent`. |
| `recover(brainIds)` | Restores deadlines without loading a neural model. Returns successful IDs and per-instance errors. |
| `undelivered(brainId)` | Returns saved resolutions that still need durable host delivery. Suspended copies are excluded. |
| `acknowledge(brainId, questionId, resolutionId)` | Marks a specific resolution delivered, after its durable conversation/action receipt exists. Host-only. |
| `waitForResolution(brainId, questionId, signal?)` | Optional asynchronous wait. Aborting the waiter preserves the question. |
| `resume(brainId, questionId?)` | Explicitly activates copied requests. Pending requests receive a fresh configured wait interval. |
| `close()` | Stops timers/subscriptions and rejects live waits; saved questions and resolutions remain. |

Responses use `{optionIds: ["intro", "examples"]}` or `{text: "My answer"}`. An Other answer is an explicit written alternative, not a way to send an option ID that was never offered. The host validates every selection, text answer, ownership ID, and saved resolution.

## Persistence and portability

State lives under `<instance>/questions/`:

- `preferences.json`: format/version, user timeout settings, and policy revision.
- `records/<question-id>.json`: the exact request, ranked options, status, timeout, resolution provenance, delivery acknowledgment, and copy lineage.

Writes sync a private temporary file and atomically replace the owned destination. Per-instance operations serialize conflicting answers and timer expiry. Listeners receive cloned events only after the write is committed. Invalid state and redirected filesystem paths report errors; they are not silently discarded. Incomplete temporary files are never treated as questions.

`src/main/userQuestionPortability.ts` exposes:

- `safeUserQuestionPath(relative)`: validates allowed saved paths.
- `savedUserQuestionFiles(brainDirectory, brainId)`: returns a validated map of instance-relative paths to source files.
- `validateSavedUserQuestionFiles(files, brainId)`: verifies imported file content and ownership without starting timers.
- `rekeyUserQuestionRecord(record, sourceBrainId, destinationBrainId, copiedAt?)`: rewrites ownership and its request checksum; retains original resolution and adds parent provenance.
- `rekeySavedUserQuestions(copiedBrainDirectory, sourceBrainId, destinationBrainId, {copiedAt?, signal?})`: rekeys an unpublished copy/import staging directory and supports retry after interruption.

Copied/imported records are suspended. Startup recovery and listing do not expire them, deliver old resolutions, or start a brain. Explicit instance Load/Resume should call `questions.resume(brainId)`. Parents are unchanged. Repository fork/export/recovery integration must include this subtree and call the rekey helper only on the staged destination.

## Native `user.ask` integration contract

The main integration should register the typed question schema with the native capability curriculum. The native action provides a prompt, mode, options, bounds, Other setting, and an optional suggested response. Main supplies instance/turn/action ownership and derives the idempotent request ID from its actual action receipt. The native schema does not expose timeout preferences, resolution acknowledgment, or approval fields.

The request returns its durable pending ID immediately so independent work can continue. On a resolution, the host records the exact chosen options/text, provenance (`user`, `timeout`, or `cancellation`), question ID, and resolution ID through the normal durable conversation/action result path. It must deduplicate by resolution ID, then call `acknowledge`. On restart, it drains `undelivered` through that same path. An assumed answer must remain labeled as an assumption during learning and continuation; it is never relabeled as a human instruction.

`src/renderer/src/UserQuestionPanel.tsx` provides the compact renderer panel:

```tsx
<UserQuestionPanel brainId={brain.id} api={window.omni.questions} onResolved={onQuestionResolved} />
```

The CSS is imported by the component. Place it outside the composer form; individual questions have their own answer forms. It subscribes before loading saved questions, merges events received during that read, and preserves draft answers when unrelated events arrive. Pending questions appear first; history and user-controlled minutes/seconds settings are collapsed. Long details, choices, and history can scroll. Native radio/checkbox/textarea controls support keyboard input and screen readers. Minimum/maximum selections are checked before sending, then validated again by the host. Timeout assumptions are explicitly labeled. A copied question exposes Resume and cannot be answered until resumed.

`onResolved` receives new durable resolution records once per resolution ID while mounted. It is not called for historical answers loaded from disk, and is a presentation callback rather than the durable learning/delivery mechanism. The host owns continuation through the acknowledged result flow above.

IPC/preload, app placement, native `user.ask`, and repository lifecycle hooks belong to the main integration lane. Stop should cancel a current wait or task as appropriate without deleting saved requests.

## Focused evidence

The 16 host/storage tests cover durable request/answer identity, recommendation order, checkbox bounds, Other/written input, idempotence, concurrency, timer expiry and restart, policy changes, copy dormancy/resume, provenance, delivery acknowledgment, corrupt input, redirected files, and event isolation. Nine renderer presentation/SSR checks cover option bounds, exact free text, stale-event ordering, countdown versus committed assumption, minutes/seconds validation, accessible form markup, paused copies, and collapsed assumption history. These checks do not construct or train a model. Integrated screenshots, live UI interactions, and the native action path remain separate integration evidence.
