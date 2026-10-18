# Learn and Fact Check — bounded UI evidence

## Scope

October 11 continuation: the actual React `LearningWorkspace` and
`UserQuestionPanel` ran in an isolated Electron window with explicitly labeled
synthetic transport. No brain was created, no training or network request ran,
and no private user source was opened. This verifies controls and presentation,
not trained understanding or the production backend's learning quality.

The desktop source mounts Learn and Fact Check as separate Run views using the
real typed `window.omni.learning`, questions, and managed dataset-preview APIs.
These views remain accessible while a foreground turn is active; the backend
owns queuing, permissions, safe update boundaries, provenance and receipts.

## Inspected flow

1. **Source selection — passed in the fixture.** Files/folder preparation shows
   correlated discovery/hash progress and Stop. A manifest representing 2,001
   files displays its selected root name and full counts, not an opaque ID.
   Selection alone does not begin learning. The controls clear selected input
   only after a confirmed task submission.
2. **Progress, Pause and Resume — passed in the fixture.** A live source event
   displays 99.0%, Pause removes the running-time estimate, and Resume restores
   the running phase without dropping progress. A source percentage is not
   reported as overall dataset completion. Completion requires the final
   traversal receipt. Screenshot `06-paused-task-footer-dark.png` shows the
   task footer, including controls and steering.
3. **Task steering — passed in the fixture.** Instruction and typed source-scope
   replacement update the selected task revision, clear the acknowledged draft,
   and preserve progress. The inspected valid scope used `allowedSites` and
   `excludedSites`. The synthetic transport is not proof of backend URL/date
   enforcement; the production coordinator validates those independently.
4. **Stop — passed in the fixture.** Keep running dismisses confirmation; Stop
   task produces Stopped with Resume still available. Committed learning/cursor
   retention is communicated explicitly. Inspection found adjacent confirmation
   buttons touching; the existing wrapping action-row style now separates them.
   The affected light-mode state was rechecked and captured in
   `10-stop-spacing-light.png`.
5. **Evidence review and correction — passed in the fixture.** A definite
   conclusion cannot save without an explanation and selected evidence. Saving
   preserves the user-review label; learning its proposed correction calls the
   correction API once, disables duplicate submission, and keeps its child
   learning task out of the Fact Check task list. No source or model proposal is
   described as guaranteed truth.
6. **Task questions — passed in the fixture.** Exactly the selected task's
   question appears while an unrelated pending question is excluded. Answering
   changes it to No open questions and retains its expandable past-question
   entry. Screenshot `08-task-question-light.png` records the waiting state.
7. **Presentation — partially verified.** Dark and light states were inspected
   at the normal window scale, including bottom-of-page content, focus and
   disabled controls. Enlarging text with the native View zoom caused the
   results layout to stack and evidence text to wrap without horizontal
   clipping (`09-review-reflow-light.png`). This was a zoom/reflow check, not a
   claimed independent physical narrow-window or all-theme audit.

## Defects corrected during this bounded run

- Live event batches were cleared before a deferred React updater read them,
  losing displayed progress. The batch is now snapshotted first.
- Source-preparation Stop could leave a stale “Stopping” message after settling;
  correlated final acknowledgement now replaces it.
- A static queued-success notice obscured subsequent task state. Submission now
  selects the task and reveals its panel; subsequent progress does not scroll
  the user away from what they are reading.
- Source labels now show human-readable selected roots with expandable full
  roots/counts; confirmed submissions clear their old input and retry ID.
- Stop confirmation buttons now use the existing spaced, wrapping action row.

## Evidence and limits

Accepted PNGs are under `test-results/learning-ui-manual/`; each was captured
through native computer use and the saved bytes inspected. Earlier captures
`04-progress-before-event-fix-dark.png` and
`05-source-99-footer-dark.png` are failure/partial-view evidence, not acceptance
of the final progress footer. The source-event regression and six other focused
presentation cases pass (seven total, 36 ms in the latest affected run).
Renderer TypeScript also passed after the spacing correction.

Still open: production desktop/backend end-to-end learning integration, broader
retention/capability qualification, physical compact-window/all-theme coverage,
full keyboard/screen-reader accessibility, and the complete v2 feature audit.
The full user training folder remains deferred.
