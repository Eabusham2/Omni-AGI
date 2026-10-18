# Host clocks, stopwatches and scheduled tasks

## Implemented scope

The desktop host has a durable operational clock workspace for each existing mind. It does not create a second brain, replace Active mode, add a persona, train a reward model, or pretend that a saved reminder is neural memory. A due wakeup submits its exact saved task through the normal same-brain chat/action controller. Normal working-memory/learning and tool permissions then apply.

Source: `hostClock.ts`, `clockStore.ts`, `clockService.ts`, `clockRuntime.ts`, `clockPortability.ts`, `clockCapabilityPack.ts`, shared `clock.ts`, and `ClockWorkspacePanel.tsx`/CSS. Tool declarations live in `tools/catalog.json`.

- `system.time.now`: actual UTC date/time, local representation, selected IANA time zone, offset, monotonic sample and process-session identity. The UI uses the same host source; its live-time endpoint performs no model or SQLite read.
- `studio.stopwatch`: start/status/list/lap/laps/pause/resume/reset plus actual native output-rate history. Rates come from measured native token counts and elapsed time, never prose length or a decorative fixed multiplier.
- `studio.schedule`: create/list/status/edit/pause/resume/cancel, paged occurrences and exact occurrence inspection, reviewed continuation, and typed event signals.
- Trigger choices: relative timer, explicit-offset calendar instant, or a matching named event with an idempotent source-event ID.
- Delivery choices: a host notification, or the exact saved task through the ordinary chat controller. Task text is not augmented with a hidden behavioral prompt.
- Optional fixed-duration repeats use a visible coalesce-or-skip missed-time policy. Coalesce produces one due occurrence rather than replaying an unbounded backlog; Skip waits for the next future interval.

The Run Schedule view is accessible while unloaded and while other work continues. It includes creation/editing, Load-when-due, stopwatches/laps, saved tasks and receipts, current access review, pause/resume/cancel and paged measured speed. Active-mode choice uses the existing Active control; this is not a second autonomous scheduler.

## Time, restart and sleep

Stopwatch duration and laps use monotonic time. A hard restart cannot reconstruct the old process's monotonic origin: the watch pauses at its last durable measurement and explicitly excludes the unmeasured gap. Normal Quit checkpoints the current measurement. The operational cadence checkpoints active watches every five seconds instead of writing every display frame. Display interpolation does not slow the model or manufacture measured laps.

Timers use monotonic duration while the process is awake, so an ordinary wall-clock correction does not prematurely expire them. Electron's actual suspend/resume events reconcile elapsed sleep omitted by a platform's monotonic clock; the stopwatch itself remains an honest monotonic measurement. Restarted timers recover their persisted wall deadline. Calendar instants remain absolute: pausing/resuming does not silently move an appointment to a different time. UTC/offset and IANA zone are retained separately, including daylight-saving display.

These behaviors use the documented [Node high-resolution clock](https://nodejs.org/api/process.html#processhrtimebigint), [Node SQLite transactions](https://nodejs.org/api/sqlite.html), and [Electron suspend/resume events](https://www.electronjs.org/docs/latest/api/power-monitor). This is not an OS boot/wake service. Studio must be running to dispatch; if closed or asleep, its saved missed-time policy applies when it returns. Existing background-residency preferences remain separate.

## Load, permissions and exactly-once ownership

Default: an unloaded mind stays unloaded and its due task remains visibly pending. A schedule may explicitly enable **Load this instance when due**. That option is part of the reviewed create/edit arguments and durable pre-effect intent; it never changes global Auto-load or grants a tool permission. Normal `loadInstance()` resource admission is used. Fit failures stay pending with a bounded retry interval, not a crash, a fallback model, or a resource bypass.

Each occurrence captures its exact task/hash, schedule revision, permission revision, Load option, fixed turn ID, trigger/event ID, due time and eventual state. Editing a schedule changes future occurrences; it does not silently rewrite a pending task or widen that task's Load authorization. A changed/revoked policy requires explicit current-access review. Policy and cancellation are rechecked around Load and before dispatch.

Before dispatch, the main-only `queryNativeTurnStatus()` verifies the exact brain/turn/input without loading a model:

- Committed: recover its actual receipt, never Load or resend it merely to obtain the result.
- Retained: reattach the authenticated existing cursor through the normal controller.
- Active: wait for the original attached turn; do not start another.
- Accepted but unfinished without a resumable cursor: retain for review, never replay effects.
- Absent: allow the first dispatch only; an already claimed but unverifiable attempt remains review-required.

Busy work queues without Steer/reset. Other retained turns keep their normal priority. Shutdown freezes new wake admission; supported in-flight work uses the existing native checkpoint lifecycle. Cancel affects future/pending occurrences, not already completed effects or a running chat's independently controlled Stop.

Notification dispatch has an at-most-once durable claim. The OS may suppress a banner, or a crash may occur between that claim and the OS call; the UI therefore says notification **requested**, not guaranteed delivered. An OS notification failure remains visible. This does not claim universal exactly-once execution of arbitrary third-party effects; those still require their own normal tool receipts and recovery boundaries.

## Persistence and portability

`clock/workspace.sqlite3` stores schedules, occurrences, command idempotency, laps and actual rate history. It uses short FULL-sync SQLite transactions, bounded pages and an owned app directory, never source paths supplied by a server. This small operational store uses a DELETE journal so short-lived read connections do not recreate WAL/SHM files on every poll. Stored row counts are not limited by the IPC page size.

The existing host workspace archive includes this exact database in snapshots, forks, exports/imports and recovery. Hashes, integrity, owner, task bytes, lap sequence and rate/count relationships are checked. A new-owner copy pauses schedules and watches and retains pending task/effect receipts for review. Historical command results remain byte-exact in inherited storage, without becoming new-owner execution authority. Same-identity recovery retains its command idempotency keys. No credential or permission is inherited from these operational records.

Full merge projects clock state into the new owner. Proven identical ID/content variants deduplicate; divergent task/lap variants remain separate. Original parents and raw archived state remain retained. Known inherited native turn IDs are rebound to the authenticated merged execution queue. Event-ID conflicts retain source receipts and refuse an ambiguous new signal rather than triggering duplicate effects.

## Tool teaching

`host-clocks-scheduled-wakeups-1` is a separately versioned declared capability pack. It covers every offered clock/schedule action and explicit Load choice. All examples are marked project-authored declarations with `executedHostActions: false`; placeholders are not fake successful executions. Initial Build teaches it before first chat, and existing loaded native minds use the existing supplementary capability mechanism. Frozen origin-v3 records are not changed. No readiness quiz, hidden system prompt, foreign teacher or model training was run as part of these host tests.

## Recorded verification and boundaries

Scoped host fixtures, not trained-intelligence qualification:

- 17 fake-clock/real-SQLite service cases: zones, monotonic laps, restart gaps, awake clock changes, actual suspend/resume gap reconciliation, unloaded/busy/shutdown behavior, explicit resource-admitted Load, revoked access, immutable pending scope, no committed replay, repeats, pause/cancel, actual rate history and event/request deduplication.
- Six real file/SQLite portability cases: archive/adopt, existing fork hook, all 130 rows beyond a page window, unsafe extra-file rejection, idempotent merge, source-owner/input integrity, and same-owner recovery command retention.
- Two real ToolExecutor/intent/SQLite cases with mocked host edges: Ask approval binds the exact Load flag; Off/revocation stops dispatch. Terminal outcome callback is observed; no neural inference is claimed.
- One declared-pack case validates every example against the actual structural schema and confirms full action coverage, deterministic hash and a separate prior pack.
- Latest combined run: **26 cases passed**, 2.61 seconds test execution (9.79 seconds including runner startup/collection).
- After the small-store DELETE-journal refinement, the eight affected portability/tool cases passed again (2.73 seconds test execution). Node TypeScript and the repository whitespace check passed at this scoped handoff.
- Root separately reports eight duration-presentation cases and [native-Electron synthetic UI interactions/visual checks](clock-ui-visual.md), including 320-pixel light layout, top/bottom scrolling, approval/error visibility and stopwatch controls. These are synthetic host receipts, not real model capability or notification delivery.

Remaining distinct scope: packaged OS-specific notifications and power-event delivery, real resource-admitted native scheduled wake/continuation end-to-end, timezone-aware recurring local-time/RRULE calendars or external calendar providers, and the broader reusable-script/VM/market-stimulus automation system. Typed event signals are supported; this slice is not an arbitrary unattended script runner. Detailed hardware/sensor performance remains the resource-tool lane. No user-corpus training, API/provider call, CI, commit or release was performed here; the complete v2 goal remains open.

## Independent action-file extension

The subsequent R164 host implementation adds an explicit `automation`
delivery with an immutable file revision/input reference and an independent
durable run handoff. It does not overload notification/task text or change
native wake/Load semantics. Event handoff does not wait for a pending neural
wake; both delivery owners share this existing clock rather than starting a
second timer loop. The current declared Clock pack is separately versioned
`host-clocks-scheduled-wakeups-2`; the earlier results above describe the prior
pack/scope. Reusable typed action/script delegation, unknown-effect review,
per-step access and dormant portability are now described in
[host-automations.md](host-automations.md). Arbitrary VM isolation, real market
stimulus capture and universal third-party exactly-once execution remain
unproved; the earlier open native/platform qualifications are unchanged.
