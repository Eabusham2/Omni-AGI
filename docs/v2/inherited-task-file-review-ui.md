# Exact-file review before inherited resume

October13, 2026. Existing ChatWorkspace prepares reviewed merged-task files
**before** setting active-turn/output/sending ownership. An already attached
writer skips the pre-resume review RPC; it uses the existing reattachment path
without re-entering native work.

The compact existing-style card shows the authenticated task and exact file
names/types/bytes/hashes. Main approval revalidates the native frontier, sealed
closure, epochs and current file permission; parent credentials/device grants
never transfer. Leave paused does not approve or resume. Pending review retains
the unsent draft/files and queue. Late results are fenced by instance/execution/
load scope; a review error releases only its review-button busy state.

Source keeps file-dependent tasks paused while file access is Off. A pending
user question asks whether explicit no-file resume should instead be supported.
No missing answer became a hidden grant or downgrade in available evidence.

## Observed UI steps

The real review component and normal app styles use **synthetic disclosure and
callbacks**: no brain, actual file, grant, adoption, resume or provider runs.
In-app-browser JPEGs were saved/reopened from exact bytes; temporary viewport
overrides were reset afterward.

1. **Desktop light — pass.** Exact file, access explanation, collapsed task/
   identity disclosures and both decisions are visible. Draft is retained.
   ![Exact file review](/Users/eyadabushama/.codex/worktrees/9eaa/Omni-AGI/test-results/inherited-review-ui-manual/01-exact-review-light.jpg)
2. **320-pixel expansion — pass.** Task, long filename, MIME and complete hash
   wrap without horizontal overflow. The DOM exposed native summaries as
   generic elements; the initial button-selector mismatch was corrected to
   their observed text, not classified as a product failure.
   ![Expanded narrow review](/Users/eyadabushama/.codex/worktrees/9eaa/Omni-AGI/test-results/inherited-review-ui-manual/02-narrow-expanded.jpg)
3. **Paused dark/review busy — pass for presentation.** Leave paused removes
   approval and offers Review again without changing the draft. Synthetic
   verification disables only review decisions. Submissions/grants remain0.
   ![Paused review](/Users/eyadabushama/.codex/worktrees/9eaa/Omni-AGI/test-results/inherited-review-ui-manual/03-paused-dark.jpg)

Two new static presentation cases passed87ms; Web TypeScript passed after App
integration. Thirteen actual small host-closure cases, using synthetic native
callbacks, are separately documented in inherited-task-files.md. These UI
checks do not replace them or prove real native end-to-end resume, packaged
behavior, all themes/hover states or full accessibility compliance.
