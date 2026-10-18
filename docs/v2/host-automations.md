# Independent host action files

## Contract and scope

R164 in REQUEST_VERBATIM.md and T1.5 in FEATURE_MATRIX.md ask for reusable,
timed, stimulus-driven actions that do not wait for neural decoding. The host
now interprets an explicit versioned action file through the existing
ToolExecutor. It does not load a brain, fabricate a neural tool turn, install a
hidden task prompt, replace Active mode, or introduce another model.

Source: `shared/automation.ts`, `automationFormat.ts`, `automationStore.ts`,
`automationAuthority.ts`, `automationService.ts`, `automationRuntime.ts`,
`automationResources.ts`, `automationPortability.ts`, the typed Clock delivery,
and the narrow existing IPC/repository/learning-workspace hooks.

The supported interpreter has sequential typed tool steps, delegated scripts,
timed waits, conditions, and explicit input/event/prior-result data references.
An existing shell, code, MCP or plugin adapter can be a step; it keeps that
adapter's actual current policy. `script-delegate` is a descriptive style, not
a second unpermissioned runner or a grant to call arbitrary host APIs.

## Action-file format

Importing or saving a file is inert. JSON is validated against version 1 and
stored as an immutable, checksummed `.omni-action.json` revision:

```json
{
  "format": "omni-host-action-file",
  "version": 1,
  "name": "Inspect a selected event",
  "steps": [
    { "id": "wait", "kind": "wait", "delayMs": 250 },
    {
      "id": "time",
      "kind": "tool",
      "toolId": "system.time",
      "action": "now",
      "arguments": { "timeZone": "UTC" },
      "when": {
        "reference": { "source": "input", "path": ["enabled"] },
        "operator": "equals",
        "value": true
      }
    }
  ]
}
```

- A tool argument may use
  `{"$automationRef":{"source":"input","path":["selectedPath"]}}`.
  Source choices are `input`, `event`, or `step`; a step reference also names
  `stepId` and reads only a committed actual output.
- `$automationJoin` joins scalar authored operands; `$automationLiteral`
  escapes an object that would otherwise look like a reference.
- Referenced incoming data is copied as data, never recursively reinterpreted
  as a reference, code, activation or permission.
- Conditions support presence, equality, inequality and numeric comparisons.
  Missing required operands fail visibly; they are not silently substituted.
- Waits use the host monotonic clock while resident. The OS/event loop may
  delay actual dispatch; milliseconds are not a hard real-time guarantee.

Portable files contain no brain owner, activation epoch, credential or grant.
Saving a new revision invalidates the old trigger activation. There is no
arbitrary stored-row limit; list/history page sizes bound one IPC batch only.

## Current access and review

All mutations go through `studio.automation` in the ordinary ToolExecutor:
`save`, `activate`, `disable`, `run`, `pause`, `resume`, `stop`, `approve` and
`review`. Save/activate/run/review accept a stable optional `requestKey`; the
same key cannot be reused with changed operands. Read actions include `list`,
`status`, `export`, `runs`, `steps` and bounded `current-step`.

Manual Run binds an exact immutable revision, exact input bytes, current
automation-policy revision and installation-local owner epoch. Timer/event
use additionally requires explicit activation with `allowTriggers: true`.
Every nested step separately re-enters current Off/Ask/Auto/Full checks, plugin
resolution, resource admission, authorized pre-effect intent and actual
result retention. A script's internal instructions remain the existing script
adapter's scope; this is not a per-syscall virtual-machine security claim.

Ask does not expire into permission. Both **Approve pending step** and
**Review uncertain attempt** require an exact, one-use outer human approval
even under Full/Auto. Echoing the pending nested token is not human authority.
Changed arguments or policy invalidate the approval. Unknown-effect review
also records an acknowledgement of possible prior effects, a reason, and an
explicit choice: skip, or a NEW attempt. Resume alone cannot resolve an
unknown/failed attempt or approve an Ask step.

Stop aborts what the adapter can abort and prevents future steps. It retains
already committed messages, intents and actual/partial outcomes. It cannot
undo an already performed external effect. A failed or interrupted started
operation is not automatically repeated.

## Durable execution and Clock triggers

Each run owns its exact program/input hashes, next step, activation revision,
per-attempt request IDs, pre-effect intents, terminal outcomes and a
hash-chained event history. Completed and explicitly skipped step receipts
are immutable. Actual outcome retention precedes advancing the next step and
precedes optional normal action-result learning.

The existing Clock scheduler has a distinct typed delivery:

```json
{
  "label": "Selected host action",
  "kind": "event",
  "eventKey": "source.changed",
  "delivery": "automation",
  "loadWhenDue": false,
  "automation": {
    "definitionId": "selected-action-id",
    "revisionSha256": "<exact saved SHA-256>",
    "inputs": { "enabled": true }
  }
}
```

Timer and explicit-offset calendar triggers use the same reference. They do
not contain a disguised task prompt. Existing notification/wakeup semantics
and the wakeup-only explicit Load option remain separate. Typed event signals
request immediate host handoff; the action delivery owner is independent of a
pending native wake, without adding a second timer loop.

A due occurrence hands off idempotently to one deterministic action-run ID.
`delegated` means that durable run accepted ownership, **not** that its effects
completed. A lost handoff acknowledgement recovers that same run rather than
creating another. Unique source-event IDs and their data hashes prevent a
repeated signal from creating duplicate occurrences; conflicting reused IDs
are rejected. Pending occurrences retain their exact original reference when
a schedule is edited.

## Pause, restart, copies and recovery

Normal Quit pauses future steps and checkpoints the remaining timed frontier,
then waits for the current actual receipt. If an effect does not acknowledge
within 30 seconds, Quit is refused; the app remains open for inspection/Stop.
Reopening admission after refused Quit does not automatically resume paused
runs. A hard restart retains a started intent without a terminal outcome as
unknown/review-required, not as proof that nothing happened.

Fork/export/import/recovery include the action database, immutable program
revisions, inputs, outcomes, command history and progress through the existing
complete learning-workspace archive. Copies and restored definitions are
dormant. Unfinished copied work needs human review. Installation-local epochs
are not exported; replacement rotates the epoch so an older enabled snapshot
cannot resurrect old authority. Same-identity recovery preserves command keys
and retains later actual effect histories as dormant branches rather than
pretending that restoring neural state undid those effects.

Full merge preserves parent files. Proven identical ID/content ancestry
deduplicates; divergent operational histories remain separate. Only typed
host definition/run references are projected into merged Clock records.
Historical invocation/output bodies remain unchanged, including user-owned
keys named `brainId`, `runId` or similar. Copied historical action occurrences
cannot be resumed as fresh effects; review the action file and create a new
trigger instead. No credential, approval token or live external handle is
inherited as authority.

## Resources, teaching and evidence boundaries

Production admission checks the selected app RAM ceiling, current owned
Electron/main-worker RSS, physical/heap parse headroom and disk-write reserve
before the next step. These are cooperative checks, not an OS-enforced hard
RSS sandbox for arbitrary subprocesses, plugins or remote services. Adapter
timeouts/cancellation and platform limitations remain truthful. Studio must
be running; no OS boot service or market reaction-time guarantee is claimed.

`independent-host-action-files-1` is a separately versioned declared capability
pack. The expanded typed Clock schema uses `host-clocks-scheduled-wakeups-2`.
The examples explicitly declare `executedHostActions: false`, contain no
reward/persona prompt, and do not modify frozen origin-v3. Initial Build and
existing supported supplementary teaching use their real native receipt
paths; no model was trained or loaded to test this host subsystem.

Scoped checks cover real SQLite/immutable files and the actual ToolExecutor
with labeled mock host effects: exact operand data, conditions, Ask/Stop,
one-use approval, revocation and partial outcomes, resource refusal, unknown
restart review, lost-handoff recovery, native-independent event delivery,
unsanitized portability, dormant merge references, source isolation and
corrupt/unsafe inventories. Root owns the separate renderer/UI acceptance.

Recorded verification, not trained-model qualification:

- Combined affected run: 55/55 cases across ten host files passed. It took
  301.54 seconds; the existing full-merge host file accounted for 294.62
  seconds, so that slow file was not repeated.
- Subsequent isolated automation/declared-pack checks: 15/15 passed after
  adding stable UI request keys, atomic Pause admission and the explicit
  human-approval boundary. The four automation files then passed 21/21 after
  bounded metadata-read and closed SQLite-schema admission were added.
- The new imported-trigger rejection and current exact-schema guard passed
  one targeted case; the two affected service/portability files passed19/19.
  Final Node TypeScript and whitespace checks passed at this handoff. No old
  native Load/curriculum fixture was repeated.
- Root separately reports eight presentation/draft checks, Web TypeScript,
  and [synthetic native-Electron UI checks](action-plugin-ui-visual.md),
  including pending step75, outer/nested approval, schedule edit/decline and
  explicitly acknowledged Skip. Synthetic host gateways do not prove real
  provider cancellation or native learned tool authorship.

Remaining distinct qualifications: real OS/provider/plugin cancellation and
third-party effect deduplication, packaged Windows/Linux process measurements,
arbitrary VM/sandbox adapters, actual stimulus capture within third-party apps,
and the future trained brain's ability to author/use these tools. This is not
a universal exactly-once transaction across arbitrary third-party APIs. No
financial, browser or device action was executed in these fixtures. No CI,
commit, push or release was performed by this lane.
