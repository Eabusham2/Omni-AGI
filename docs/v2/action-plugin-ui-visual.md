# Action-file and plugin controls — bounded visual check

October 11, 2026. This checks the real React components in one Electron window
with explicitly synthetic APIs. No model, permission grant, plugin installation,
real file read, schedule or external effect ran. Backend SQLite/ToolExecutor
evidence is separate in [host-automations.md](host-automations.md) and
[plugins.md](plugins.md).

## Scope and findings

1. **Action-file entry — readable in the captured dark desktop state.** Saved
   files, run status and receipts are discoverable without creating a run.
   `01-action-start-dark.png` records the initial view. The file-count grammar
   was subsequently corrected. This is not a complete Build/chat audit.
2. **Pending nested request — corrected and checked at 320 CSS px, light.**
   A current pending receipt at step 75 is available through the dedicated
   `currentStep()` read, without traversing the older history page. Its real
   tool/arguments are primary; control IDs/tokens and the full retained receipt
   are collapsed. The first verbose dark layout (`02`) was improved, not kept
   as the accepted final design. The synthetic approved request settled to
   complete and removed its pending controls; this proves UI registration only.
3. **Plugin learning event — live status update verified.** A synthetic
   pending→complete event refetched the exact brain/plugin/hash record while
   preserving an unsaved instance-enable checkbox. This was observed in AX and
   `04-plugin-learning-320-light.png`; its vertically separated checkboxes were
   then corrected. No actual native learning was performed by the fixture.
4. **Plugin access revision — stale edits discarded explicitly.** A different
   actual access revision replaced the draft with current settings and showed a
   review warning. Corrected checkboxes remain beside their labels at 320px;
   warning text is readable on the captured light surface. The event does not
   auto-scroll the user away from reading. No access was granted by the fixture.
5. **Action-file schedule edit — exact typed request checked.** The UI submits
   `delivery: automation`, the exact file hash/inputs, saved schedule revision,
   explicit `loadWhenDue: false` and blank task. Create-only `kind`/`eventKey`
   are not sent to the edit endpoint. A declined review disappears with an
   honest declined status. Hovered approval text remains readable in the light
   capture. No schedule was changed outside the fixture.
6. **Unknown-effect recovery — acknowledgement and reason required.** Both
   choices are disabled until the user acknowledges possible prior effects and
   provides a reason. Opening the review focuses/reveals the new controls.
   The outer review still asks for a decision. A misleading label that showed
   the previous tool as a new request was corrected: Skip is now explicitly
   “do not run it again,” and a new attempt warns about repeated effects and
   the need for explicit Resume. Previous receipts remain inspection evidence.
7. **Best Work settings — previously open control check passed.** At 320px
   light, Enable and the budget remain editable during a synthetic foreground
   turn while Apply is disabled with a local explanation. After that turn ends,
   a typed17 stays invalid/disabled;16 enables Save. Saving produces exactly
   `{bestWorkEnabled: true, bestWorkExplorationBranches: 16}`, shows Saved and
   disables the now-unchanged Save button. The receipt explicitly says
   `modelExecution: false`; no candidate sampling or preference training ran.
8. **Call/conversation layout — wrapper reflow checked.** The new real
   `CallLayout` puts controls beside a separate conversation at desktop width
   and stacks them at320px. Its two source cases keep the capture-owner children
   mounted when controls are hidden. Captures09/10 use explicitly synthetic
   control/conversation children (including a bare fixture textarea), not the
   production composer or microphone/camera controller. They establish layout
   placement only, not active capture, preview, speech, barge-in or final Call
   acceptance. Instance speech routing and genuine raw-audio input remain active
   implementation lanes.

## Accepted final-state captures

These exact saved PNGs were reopened and visually inspected. They are a 320px
component within a wider Electron window, not proof of a physical 320px device.

![Step 2: readable pending nested request](../../test-results/action-plugin-ui-manual/03-pending-review-320-light.png)

![Step 4: current access, inline checkbox labels and stale-edit warning](../../test-results/action-plugin-ui-manual/05-plugin-conflict-320-light.png)

![Step 5: exact saved schedule edit and readable hover](../../test-results/action-plugin-ui-manual/06-action-schedule-edit-320-light.png)

![Step 6: explicit skip decision, not a fabricated new tool request](../../test-results/action-plugin-ui-manual/07-explicit-skip-review-dark.png)

![Step 7: valid Best Work settings saved, no model execution](../../test-results/action-plugin-ui-manual/08-best-work-320-light.png)

![Step 8: actual layout wrapper with synthetic children, desktop dark](../../test-results/action-plugin-ui-manual/09-call-layout-wide-dark.png)

![Step 8: stacked wrapper with synthetic children, 320px light](../../test-results/action-plugin-ui-manual/10-call-layout-320-light.png)

## Source and bounded checks

- `TimeWorkspace.tsx` mounts the collapsed action workspace beside the existing
  Clock workspace. Its explicit edit callback routes automation schedules into
  their typed editor rather than casting them to a neural wake task.
- `AutomationWorkspace.tsx` uses read-only preload inspection and the normal
  `tool.execute` mutation path. It retains exact approved operands, uses stable
  request keys, clears changed reviews, refreshes opened receipts through events,
  and offers paged history without a stored-row ceiling.
- `PluginManager.tsx` uses exact-scope event refresh, stale response guards,
  same-revision semantic-only draft merging and hash/revision-bound global
  Enable. Event listeners unsubscribe; there is no persistent status poll.
- Five new action-presentation and three access-draft checks pass. Renderer
  TypeScript passes after the final changes. These focused checks overlap no
  claim about real provider actions or trained quality.
- The React review covered conditional hooks, independent parallel reads,
  event cleanup, per-owner response guards, labels, native controls and
  user-triggered focus/reveal. The UI-audit checklist caught the noisy approval,
  narrow checkbox reflow, light warning contrast and recovery-label ambiguity.

## Explicit limits

The fixture has no persistent API state; HMR may recreate its synthetic data.
This is not restart/durability proof. Complete manager import/export/maker
interactions, every theme/layout, keyboard-only/assistive-technology acceptance,
production Time/Clock callback flow, real provider cancellation and packaged
cross-platform behavior remain separate checks. No broad model/corpus run,
commit, push, CI or package release occurred in this UI slice.
