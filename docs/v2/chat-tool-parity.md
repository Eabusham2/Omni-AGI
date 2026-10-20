# Retained API teaching and host statistics in chat

October20 source/focused-contract record. These are two genuine gaps found
after the earlier documentation reconciliation, not a repeat of finished UI
or trained-model tests. No application, provider, brain, Load, training or CI
was run for this slice.

## API teacher

`data.distill-api` now exposes typed providers, discover-models, start, list,
status, pause, resume, stop and resolve-uncertain actions through the existing
`ToolExecutor`. Its adapter calls the same `ApiTeacherTrainingService` used by
the existing panel. It adds no provider, model fallback, endpoint override,
credential setter or alternate learner. Unknown credential/owner/request-ID
fields are refused before the tool intent/audit boundary. Provider status
does not disclose credential hints.

New policy defaults are Ask. Auto still asks for start/resume/model discovery;
uncertain Retry/Skip always requires an exact human approval even under Full.
Resolution leaves the job paused; it does not dispatch or implicitly resume.
Current host-owned permission/origin and resource guards survive the enqueue
return and run before provider and learning boundaries. Policy/origin retirement
interrupts only tool-owned jobs through the existing controller. Explicit panel
calls retain their existing behavior.

A guard refusal before the actual call leaves a pending, paused lesson, not a
fictional unknown provider request. A received provider ACK remains committed
if permission is subsequently lost; a newly authorized Resume consumes it
without resending. Cancellation during an actual in-flight request retains
uncertain-review custody. It cannot claim remote cancellation, and no unknown
provider or learning effect is automatically retried. Live callbacks/signals
are never stored or exported as authority.

## System statistics

`system.stats.snapshot` is a separate, ordinary permissioned read. Main supplies
the exact saved owner directory and cached device metadata, never a generated
path or shell command. It admits bounded metadata headroom and calls the
read-only `systemStatsService` with the current permission guard and Stop signal.
The helper reports observation times, sources/scopes and actual unavailable
readings; its own detailed evidence belongs to the helper's focused report.
This is on-demand metadata, not background polling, model benchmarking,
protected sensor access or a new device grant.

## Initial availability and evidence

Both protocols are registered in the native typed catalog and default-policy
list. Build's actual master-access projection includes them: Off stays Off;
ordinary access is Ask; Full requires that new explicit selection. A missing
entry in an older instance does not inherit another tool's Auto/Full authority.

The separate `api-teacher-system-stats-1` capability pack derives its schemas
from the real catalog, with declared usage examples and no executed outcomes.
The owned initial-capability path and ordinary supplementary preparation use
the existing native learning commit API. Frozen origin-v3 is unchanged; no
behavior prompt or string/regex action router is added.

- Nine distinct new inert route/protocol cases passed across scoped runs:
  catalog/native schemas, actual Build projection, initial teaching dispatch,
  credential-field refusal, Off/Ask/Auto and detached policy, exact uncertain
  review, post-revocation read refusal, discovery Stop and stats policy.
- Four new tiny SQLite/transport/learner-fixture authority cases passed across
  scoped runs. The first three held-state assertions used the wrong generic
  RuntimeJob state; corrected assertions inspect its exact durableState report.
  Production state projection was not changed. No old suite was rerun.
- One additional strict-enum validation case passed after review: nested arrays
  cannot be coerced into accepted modality or uncertain-decision strings.

These checks establish typed dispatch and custody, not successful learned tool
selection, provider/account interoperability, actual sensor/hardware readings,
parameter learning, current all-theme UI, packages or release acceptance.
