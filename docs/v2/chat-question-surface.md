# Ordinary Conversation questions

## Confirmed gap and source scope

`UserQuestionPanel` previously appeared only in `LearningWorkspace` while the
ordinary native `studio.question` tool already created durable questions.
`ChatWorkspace` now mounts that same panel immediately above its composer with
the conversation presentation. No alternate application or question backend
was introduced.

The section renders nothing while its current owner has no question, history
or service error. Its event subscription remains active. New nonsuspended
pending questions and service errors reveal and expand it; the user can close
it without changing the question. Past answers and settings use the existing
collapsed history/settings controls and existing bounded scrolling styles.
Learn and Fact Check sources remain in their task-ID-scoped workspace panel,
not a second general Conversation tray.

The React child identity binds brain, API object and presentation. Retiring a
child closes its event subscription and fences late list/response/preferences
work. Events, scans and response records must belong to the current brain;
answer/skip receipts must also retain the exact requested question ID. This
does not cancel an answer already committed by the host, substitute another
answer or replay an external effect.

## Actual host delivery semantics

`ToolExecutor` supplies `source.kind = brain`, its existing main-owned action
ID and the trusted `ToolExecutionContext.chatTurnId`. Generated `source`,
`brainId` and `requestId` values do not override those identities. Ask returns
the saved pending question. The existing `status` action can inspect its current
owned record; Ask does not call `waitForResolution` or hold a decoder open.

`UserQuestionDelivery` subscribes to resolved/cancelled records, stages their
exact completed outcome through `BrainService.queueCompletedToolExecution`,
then acknowledges the question after that sink has accepted durable evidence.
Existing selected-owner background learning governs later native consumption.
Learn/Fact Check additionally reconcile their dependent task/claim through
their existing coordinator. This change adds no forced chat continuation,
behavioral prompt, direct native waiting-action wakeup or permission authority.

Answers and assumptions remain `task-choice-only` with
`permissionsGranted: false`. Timeout stays off by default, with20 minutes as
its opt-in default. Copied/suspended questions still require explicit resume.

## New focused contracts

Five new cases passed in the existing selected files:

- `userQuestionPresentation.test.ts`: exact instance/task/conversation filtering.
- `userQuestionPanel.test.tsx`: empty hiding, real pending reveal, event-during-
  scan preservation, API replacement/late-answer retirement, Retry and wrong-
  question receipt refusal, using an inert React hook/API harness.
- `userQuestions.test.ts`: the actual ToolExecutor dispatch case and tiny real
  filesystem question service override forged ownership, retain the trusted
  turn, and send the later task-only answer to the existing evidence sink.

The focused selection passed5 cases with20 older cases deliberately skipped
in2.76s (51ms test bodies). The hook harness is not browser rendering, and the
delivery sink is inert; it proves the declared source boundary, not native
learning, remote services, model output or packaged behavior. No brain was
created, loaded, trained or queried.

## Bounded renderer fixture

`tests/e2e/chat-questions-fixture.html`/`.tsx` mount the actual panel with
explicit in-memory synthetic APIs. Controls emit Conversation, Learn or Fact
Check records; single/multiple/written choices; a paused copy; current errors;
and a held answer followed by API/brain replacement and late release. Compact
and light/dark controls exercise the existing styles. Static placement markers
are not generated messages or a complete App/IPC/backend test.

`chat-questions-main.cjs` requires a dedicated `OMNI_QUESTIONS_UI_DATA` profile
and uses the existing fixture server on port5189.

## October18 bounded native-renderer inspection

Root launched the isolated Electron fixture, not a brain. Every saved capture
was reopened from its exact bytes before acceptance. The local capture root is
`test-results/chat-questions-ui-2026-10-18/`; no screenshot is model output.

1. **Empty and pending:** no permanent empty row (01); a native-source multiple
   choice revealed the actual panel. One selection kept Send disabled; two
   enabled it (02). Tab navigation and Return submitted the selected answer.
2. **History and written input:** actual history retained both selected labels.
   Compact390px content and dark appearance were inspected (03/04), including
   editable written input, visible focus and the page bottom. This is constrained
   content inside a1060px window, not a390px native-window claim. The fixture's
   first non-scrolling outer container was corrected; its hot reload reset only
   synthetic fixture state, not app or brain state.
3. **Retirement and scope:** a held answer was released after API replacement;
   the new empty owner did not gain its question/history. A Learn-source question
   did not duplicate into Conversation. These transitions were verified through
   native accessibility state, not a claim of native answer consumption.
4. **Error recovery:** a real compact rendering defect split Retry across lines
   (05). The button now retains its intrinsic width; the affected capture is
   readable (06), and Retry cleared the synthetic error without restoring an old
   owner. This source change did not replace the error with invented success.
5. **Paused and scroll:** copied question choices remained disabled (07).
   Nested scroll containment initially trapped the bottom control (08); the
   conversation presentation now uses its one outer content scroller, retaining
   independent task-panel behavior. Resume became reachable (09), enabled choices
   only after its explicit click, and switching instances removed the old panel
   (10). Earlier diagnostic frames are not proof of the corrected bottom state.

The owned Electron preview exited0. No actual conversation, main IPC question
request, neural learning, provider, timer assumption, credential or device grant
ran. This checks the actual component with a synthetic API, not full App/backend
delivery, all themes/window sizes, accessibility conformance or packaged proof.
