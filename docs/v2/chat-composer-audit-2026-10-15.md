# Chat composer and queue — October 15 bounded UI check

## Scope

The existing production `ComposerSubmitControl` and `ChatQueueStack` were
mounted in a no-preload Electron fixture, not a replacement interface. All
callbacks were synthetic. No brain, model, training, provider, real chat
submission, task file access or neural mutation ran. The fixture kept one
unsent draft and initially three queued entries.

## Observed steps

1. **Idle, healthy:** Send was present and Stop absent. The initial capture
   `01-idle.jpg` was diagnostic only because the fixture's outer
   scroll setup cut off its bottom. The fixture was corrected before the
   following interactions; this is not an App scrolling fix or acceptance.
2. **Queue edit/reorder/remove, healthy:** the second entry was expanded and
   edited, moved earlier, and only the selected third entry was removed.
   Two entries remained in the correct order. Send now invoked its exact
   synthetic entry callback. See `02-edit-reorder-remove.jpg`.
3. **Busy click routing, fixed defect:** a cached Ctrl/Cmd display flag could
   survive a shortcut without key-up. Production previously also used that
   cached flag to route clicks, allowing a normal click to steer. Routing now
   uses the actual click's Ctrl/Meta flags only; pointer/click events refresh
   the label. An ordinary click invoked Queue, and a Command-click invoked
   Steer, each preserving the draft. One new deliberately stale-state source
   case passed (`composerModifierAuthority.test.tsx`, 8 ms). A queued-entry
   Steer invoked its exact entry callback separately.
4. **Stop pending, healthy presentation:** the synthetic Stop callback made
   Stop/Queue disabled with a Stopping label. See `03-stopping.jpg`. This does
   not prove cancellation of a real worker.
5. **Finished, healthy presentation:** selecting the fixture's finished state
   restored Send, removed Stop and changed queued-entry actions to Send now.
   The edited queue and unsent draft survived. See `04-finished-dark.jpg`.
6. **Bottom scrolling, healthy within fixture:** actual scrolling exposed the
   callback receipts and explicit zero host/provider/neural counts. See
   `05-callbacks-bottom.jpg`.
7. **Compact light, healthy within inspected controls:** native macOS
   Top Left resizing produced a 1444×962-pixel capture (not a claimed CSS
   viewport). Queue text truncates without covering the controls; expanded
   text remains editable. The completed Send and bottom receipts remain
   accessible by scrolling. See `06-finished-light-compact.jpg` and
   `07-light-compact-queue.jpg`.

Every accepted capture above (02–07) was saved and reopened before acceptance.
They are under `test-results/chat-composer-audit-2026-10-15/`.
The owned Electron window was closed normally and its process exited 0.

## Limits and remaining checks

- The fixture deliberately retains an entry after Send now/Steer so callback
  identity can be inspected. It does not prove host queue consumption,
  recovery, off-page generation or main-process sequencing.
- The App's composer key handler, applied-steer strip, IPC and actual neural
  continuation were not exercised here. Earlier evidence remains separate.
- A missed key-up can leave the display hint stale until a pointer/key/focus
  event refreshes it. It can no longer authorize a steering action by itself.
- Screenshots/AX observations establish these controls' inspected appearance
  and labels, not complete accessibility or all-theme/platform acceptance.
- Full learned-capability qualification and the explicitly deferred new-brain
  merged-resume proof remain deferred; this check does not replace them.
