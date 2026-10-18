# Best-effort operating-system work lease

Internal source evidence, October 15, 2026. No real power setting, process
priority, neural worker or model was changed during verification.

`OsWorkLeaseService` uses main-owned observations of running/cancelling engine
requests, runtime/API-teacher jobs, storage operations and running chat actions.
Queued, proposed, approval-waiting and terminal work does not acquire a lease.
Concurrent observations have independent identities/references; the final
release restores captured state. Explicit host leases are also available.

Main composes Electron's
[`prevent-app-suspension`](https://www.electronjs.org/docs/latest/api/power-save-blocker)
blocker, which permits the display to sleep. Supported owned processes get
[Node's above-normal scheduling priority](https://nodejs.org/api/os.html#ossetprioritypid-priority)
only if that improves their existing priority. Access denial is reported,
not bypassed. No admin escalation or arbitrary renderer-selected PID exists.

Actual supervisor spawn events refresh newly owned workers. Worker-close retires
the exact PID without another OS write. Restoration rechecks current ownership
so an externally reused PID is not modified. Original values are captured per
owned lifecycle, not inferred from a platform default. Long user-derived labels
cannot throw through the work-observation event dispatcher.

`window.workStatus()` is a read-only diagnostic through trusted IPC/preload.
It reports not-attempted/unsupported/denied/partial/error/restored outcomes and
the actual owned blocker/target counts. Closing the app awaits work checkpoints
before releasing this service. Restoration failures stay visible and are retried
on Close; they are not labeled successful.

This is not RAM pinning, an OS hard memory cap, a background execution entitlement
or a guarantee against forced termination, OS updates, battery policy, thermal
throttling or hardware failure. Real OS/platform behavior remains unverified.

## Evidence

- 5 new injected fake-OS cases cover reference counts, denial/unsupported state,
  running-only observations, new workers, observer isolation, restoration failure,
  long labels, preserved higher priority and owned PID reuse.
- 1 new fake-child supervisor case passed5ms: actual superviseChild spawn/close
  callbacks refresh and retire ownership without spawning a process.
- 1 new fake-preload case passed32ms: status is read-only and has no process/
  power mutation interface.

The source composition includes engine, runtime/API teacher, action, storage,
worker-start/close and shutdown hooks. Local teacher/media compute still uses
its existing managed engine/host/action boundaries; no independent child is
claimed covered merely because it exists. Full runtime/platform acceptance is
separate from these focused source contracts.
