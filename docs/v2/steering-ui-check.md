# Compact steering in the existing chat

October 13, 2026. Bounded inspection of the production `ChatDirectionStrip`,
`useChatDirections` and correlation helper with a labeled **synthetic in-memory
host API**. This is not a new application, a brain run, a learned steering
proof, a persistence test or a complete UI audit.

The existing composer, queue styling, original human task and emitted reply
remain. Exact successor directions use host-owned input and human projections;
a predecessor is never hidden because its output was steered. Dismissal hides
display, not accepted input or learning. Revision creates new text; it does
not rewrite a receipt. Unsent drafts and selected files remain separate.

## Captured steps

Captures came from the in-app browser and were saved as JPEGs and reopened
from those bytes before acceptance. Viewport overrides were reset afterward.
The fixture made zero neural submissions, learning or external tool calls.

1. **Latest entry, narrow dark — pass.** The latest pending entry is visible
   instead of starting at the oldest entry of the recent page. Original task
   and reply remain. This first capture includes only the visible viewport,
   not the offscreen composer.
   ![Narrow dark latest entry](/Users/eyadabushama/.codex/worktrees/9eaa/Omni-AGI/test-results/chat-directions-ui-manual/01-latest-dark.jpg)
2. **Desktop dark — pass.** At the 1,000-pixel breakpoint, strip width is914
   pixels and document/viewport widths agree. Content fits beside state and
   dismissal; composer remains visible.
   ![Desktop steering strip](/Users/eyadabushama/.codex/worktrees/9eaa/Omni-AGI/test-results/chat-directions-ui-manual/02-desktop-dark.jpg)
3. **Revision — pass for presentation.** Editing/submitting calls the fixture's
   new-direction callback, preserves original content and the unsent draft.
   The callback is synthetic, not actual neural dispatch.
   ![Revision preserves draft](/Users/eyadabushama/.codex/worktrees/9eaa/Omni-AGI/test-results/chat-directions-ui-manual/03-revision-preserves-draft.jpg)
4. **Light320-pixel expansion and idle action — pass.** Full text, editable
   new direction, focus ring and button are readable and contained. After
   completion the button is `Use in composer`; clicking appends to the existing
   draft without sending.
   ![Light narrow expansion](/Users/eyadabushama/.codex/worktrees/9eaa/Omni-AGI/test-results/chat-directions-ui-manual/04-light-narrow-expanded.jpg)
5. **Receipt, dismissal and earlier paging — pass.** A completion event changes
   the same entry to Completed. Dismissal does not resurrect a duplicate human
   bubble; original task/reply remain. Earlier paging makes the first retained
   entry accessible and preserves the currently read position.
   ![Dismissal and paging](/Users/eyadabushama/.codex/worktrees/9eaa/Omni-AGI/test-results/chat-directions-ui-manual/05-dismiss-page-light.jpg)
6. **Instance isolation — pass for mounted presentation.** Switching A→B shows
   only B's direction. This is not a delayed-response race or restart proof.
   ![Instance-isolated strip](/Users/eyadabushama/.codex/worktrees/9eaa/Omni-AGI/test-results/chat-directions-ui-manual/06-instance-isolation.jpg)

## Source fixes and focused evidence

- Mounted above the existing composer, retaining the editable queue. Exact
  lookup uses the host's100-turn transport batch across all displayed turns;
  the batch is not a history limit.
- `Send now` uses the prefix-preserving Steer path. An unloaded instance keeps
  its queue. Optimistic rows share their direction metadata's source timestamp.
- Revisions do not consume unsent draft attachments or edit accepted source
  or audio identities.
- The `Voice direction` label is separate from actual revision text. Empty
  audio text opens an empty revision field, not a fabricated transcript.
- Five correlation cases passed4ms, including a new merged-human projection
  case. A later audio-label revision case passed4ms with those five skipped.
  The earlier three strip presentation cases are separately recorded, not
  repeated for this UI check.
- Renderer TypeScript passed after the affected App/strip changes. Host store,
  controller and custody evidence belongs to the R187 owner separately.
- Three later exact-query cases passed5ms:201 turns in100/100/1 batches,
  stale-owner results withheld, no subsequent stale request and current
  errors propagated without silently truncating history.
- Acknowledged Unload now uses paused/detached labels, not false failure or
  cancellation. Two label cases passed4ms. Three subsequent main-certified
  lost-stream receipt cases passed3ms (old two skipped), with later Stop or
  actual reply taking precedence over an older saved cursor.

## Limits

No real brain, persisted host history, native warm handoff, permission prompt,
microphone, model load, training, provider, packaged Electron or cross-platform
acceptance occurred. Screenshots cover ordinary light/dark at these sizes,
not all themes/layouts, hover states or full accessibility compliance. Real
queue/native steering, durable custody and broad acceptance remain separate.
