# Durable steer direction history

Steer directions are durable presentation history above the composer. Each entry identifies the exact successor input and predecessor turn, shows its actual acceptance or completion state, and can be dismissed without changing the conversation or learned state. The original human task and its published reply remain intact.

This implements the compact direction history requested in `REQUEST_VERBATIM.md` line 187 and the later instruction that revising an applied direction creates a new direction. Pending queue drafts remain editable in the existing queue. Applied source content and receipts are immutable.

## Stored source and actual states

The main process creates a direction only for a steer submission. Ordinary tasks do not become directions. The immutable source includes the original owner, successor and predecessor IDs, submitted time, normalized applied text, the original submitted text when normalization differs, and selected attachment IDs. Audio directions retain safe original capture identities and hashes, not a transcript, filesystem path, microphone lease, or reusable permission.

The state follows actual controller and service evidence:

| State | Required evidence |
| --- | --- |
| `pending` | The controller retained the direction before dispatch; this is not input acceptance. |
| `received` | The exact committed accepted-input receipt matches the applied input hash and actual human row. |
| `complete` | The service returned the saved human and assistant message pair. A typed no-reply completion also has a real saved pair. |
| `stopped` or `steered` | The actual native result or matching early boundary reports that outcome. An early yield does not manufacture acceptance or a reply. |
| `failed` | Dispatch failed before a saved reply. An error after a saved reply preserves the reply and adds `post-reply-work-failed`. |

Unloading a parked turn does not mean Stop. An acknowledged host detach of a durable native continuation preserves the last actual state and adds `saved-continuation` custody. An unfinished record found after process restart is retained as dormant `process-interrupted` history, without guessing whether native input committed during a missing host write. An explicitly authenticated same-turn continuation can passively reattach to the record and supply actual subsequent receipts; this presentation service cannot initiate that continuation itself.

Accepted attachment IDs come from the main-normalized human message, including verified inherited original task-file handles. The accepted stream and direction record preserve those IDs without changing original manifests or audio source hashes.

## Display controls and correlation

`window.omni.chat.directions` exposes only `list`, `get`, `forTurns`, `dismiss`, and `onEvent`. It has no append, edit, resend, learn, tool-execute, or grant endpoint. Main IPC handlers require a trusted sender. Dismissal uses the exact current revision and changes display state only; it neither deletes history nor reverses learning.

Pages contain at most 100 entries. `forTurns` accepts at most 100 distinct currently rendered turn IDs and includes dismissed entries, so older chat pages can still correlate their exact successor inputs. `humanMessageId` and `humanTurnId` are current host presentation aliases, separate from original accepted receipt IDs. A merged inherited turn can therefore retain its original human turn while executing under a new root turn. Only the correlated successor input belongs in the direction strip; a predecessor is never hidden merely because it ended as steered or shares the same text.

Compact events contain identity, revision, state, and an invalidation type rather than repeatedly broadcasting complete user text. Read operations check transfer memory before loading pages or input bodies. Resource refusal does not truncate history. A presentation update failure emits `unavailable` without aborting or undoing actual native computation.

## Persistence and complete custody

The versioned SQLite workspace is `directions/workspace.sqlite3`. Immutable source JSON is stored once; views and a chained event history refer to its hash. The store checks owner, declared schema, row checksums, event succession, and the committed event head. Extra schemas, triggers, unsafe filesystem entries, and changed immutable archives are refused.

The existing complete workspace pipeline includes the direction database and `directions/archives/<sha256>.sqlite3`:

- Fork, export, import, and recovery use the shared snapshot and adoption helpers. SQLite snapshots include committed WAL data. Rekeying changes current host ownership and custody only; original applied content, accepted receipts, source audio identities, and earlier event rows remain unchanged.
- Copied unfinished directions stay dormant. Import, merge, and recovery do not resend, learn, execute, or recreate a permission grant from a direction.
- Full merge projects only real host-history aliases and verifies the destination human row. Identical shared applied histories are unioned once, while genuinely divergent outcomes remain separate historical branches. Their full source event histories remain in immutable archives.
- Recovery retains newer actual direction outcomes as dormant historical branches rather than erasing them with an older snapshot. The selected snapshot remains the active native state. Recovery must detach existing host trackers before replacing their files.

The direction store is not neural memory, a fact cache, a prompt template, or dispatch authority. No runtime reads its text to answer questions, train the brain, replay an input, or authorize an external effect.

## Verification

Twenty distinct new host checks passed across three short targeted runs:

- `steerDirections.test.ts`: nine cases covering immutable source and receipts, dismissal revision checks, paging and old-turn lookup, schema/event tampering, transfer-memory refusal, actual state transitions, safe audio and App attribution, dormant restart, and duplicate tracker ownership.
- `steerDirectionPortability.test.ts`: five cases covering WAL snapshot and another fork, unchanged original source database, complete raw event archives, shared-history deduplication and divergent outcomes, chronological merge, recovery history, actual current human aliases, and changed or unlisted payload refusal.
- `steerDirectionController.test.ts`: five cases covering accepted inherited attachment IDs, genuine predecessor yield with the original task and published prefix preserved, Stop during direction persistence, acknowledged detach, and early yield without invented receipts.
- `steerDirectionPreload.test.ts`: one case covering the typed read and display-only bridge and removable event subscription.

The final Node TypeScript check and scoped whitespace check passed. These checks used synthetic native boundaries and small real host databases; they do not claim a new model run, neural recall proof, complete desktop end-to-end run, packaged release, or real device test. The repository fork, recovery, and `.omni` entry points are wired to the verified shared workspace helpers; those entire repository workflows were not rerun as part of this slice. The renderer integration and its visual evidence are owned separately by the main integration lane.

## Suspended host disposition

An acknowledged host detach now has a dedicated controller path before generic failure handling. The ordered stream reports `suspended` only when the actual unloaded native observation supplies this exact turn and input frontier. A host wait without that qualified frontier reports `detached`; neither disposition invents acceptance, a reply, Stop, or cancellation. The public `ChatHostSuspensionReceipt` contains only declared safe cursor fields, the input hash, and any actual accepted-input receipt.

The main process appends this disposition to the existing presentation-only conversation ledger. It leaves native input, learned state, task scopes, and checkpoints untouched. A new detach can append a new receipt without rewriting an earlier one. These display rows are excluded from model-facing history searches. The renderer delivery-receipt endpoint cannot manufacture a certified suspended disposition.

Already-started effects keep their real receipts and are drained through their existing memoized executions; unstarted proposals are neither dispatched nor marked failed. A display-history write refusal does not turn native suspension into failure. The generation and message presentation helpers settle controls while retaining partial output and the original input identity, and permit a newer certified suspension to replace an older stale failure label. Composer handling must recognize the typed disposition before the generic IPC rejection path, without restoring the already-retained input as an unsent draft.

Nine additional cases in `chatSuspensionDisposition.test.ts` passed across targeted runs. They cover exact versus missing/foreign frontier evidence, ordered controller disposition, unadmitted waits, actual-effect retention once, display persistence without neural changes, partial-output preservation, native no-load detach binding, stale failure replacement, and safe public projection. The earlier 20 direction checks were not repeated. The followup Node check reported no lifecycle diagnostics, but the shared tree still had two separately owned plugin-parser union errors; this followup does not claim a new whole-tree green compiler or completed composer/UI acceptance.
