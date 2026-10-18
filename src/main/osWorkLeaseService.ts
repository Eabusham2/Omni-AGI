import type { ActionEvent, BrainStorageOperationEvent, RuntimeJob } from "../shared/types";
import type { OsWorkCapabilityOutcome, OsWorkLeaseEvent, OsWorkLeaseSource, OsWorkLeaseStatus } from "../shared/osWorkLease";
import type { EngineActivityTransition } from "./engineSupervisor";

export interface OsWorkPlatformAdapter {
  platform: string;
  startPowerSaveBlocker?(): number;
  stopPowerSaveBlocker?(id: number): void;
  isPowerSaveBlockerStarted?(id: number): boolean;
  priorityProcessIds?(): readonly number[];
  getPriority?(pid: number): number;
  setPriority?(pid: number, priority: number): void;
  requestedPriority?: number;
  diagnostic?(message: string): void;
}

export interface OsWorkLeaseDescriptor {
  source: OsWorkLeaseSource;
  id: string;
  kind: string;
  label: string;
}

interface LeaseRecord extends OsWorkLeaseDescriptor { references: number; startedAt: string }
interface PriorityRecord {
  pid: number;
  original?: number;
  applied: boolean;
  satisfied: boolean;
  denied: boolean;
  error?: string;
  restoreFailed?: string;
}

const ID = /^[A-Za-z0-9][A-Za-z0-9:._-]{0,255}$/;
const ACTIVE_JOB_STATES = new Set(["running", "cancelling"]);
const ACTIVE_ENGINE_STATES = new Set(["running", "cancelling"]);
const ACTIVE_STORAGE_STATES = new Set(["running", "cancelling"]);
const SOURCES: readonly OsWorkLeaseSource[] = ["engine", "runtime-job", "storage-operation", "build", "host"];
const LIMITATIONS = [
  "This best-effort lease cannot prevent forced termination, operating-system updates, battery policy, thermal throttling, or hardware failure.",
  "Priority applies only to currently owned host and worker process IDs exposed by the main process; unsupported or denied targets remain unchanged."
] as const;

function text(value: unknown, label: string, maximum: number): string {
  if (typeof value !== "string") throw new Error(`${label} is invalid.`);
  const clean = value.replace(/\0/g, "").trim();
  if (!clean || clean.length > maximum) throw new Error(`${label} is invalid.`);
  return clean;
}
function failure(error: unknown): string {
  return (error instanceof Error ? error.message : String(error)).replace(/[\r\n\0]+/g, " ").slice(0, 500);
}
function denied(error: unknown): boolean {
  const code = error && typeof error === "object" ? String((error as { code?: unknown }).code ?? "") : "";
  return code === "EPERM" || code === "EACCES" || /(?:permission|not permitted|access denied)/i.test(failure(error));
}

export class OsWorkLeaseService {
  private readonly leases = new Map<string, LeaseRecord>();
  private readonly listeners = new Set<(event: OsWorkLeaseEvent) => void>();
  private readonly priority = new Map<number, PriorityRecord>();
  private blockerId?: number;
  private closed = false;
  private sleepOutcome: OsWorkCapabilityOutcome;
  private sleepDetail: string;
  private priorityOutcome: OsWorkCapabilityOutcome;
  private priorityDetail: string;

  constructor(private readonly adapter: OsWorkPlatformAdapter, private readonly now: () => Date = () => new Date()) {
    const sleepSupported = Boolean(adapter.startPowerSaveBlocker && adapter.stopPowerSaveBlocker && adapter.isPowerSaveBlockerStarted);
    const prioritySupported = Boolean(adapter.priorityProcessIds && adapter.getPriority && adapter.setPriority &&
      Number.isInteger(adapter.requestedPriority) && Number(adapter.requestedPriority) >= -20 && Number(adapter.requestedPriority) <= 19);
    this.sleepOutcome = sleepSupported ? "not-attempted" : "unsupported";
    this.sleepDetail = sleepSupported ? "Sleep prevention has not been requested." : "This host does not expose a supported app-suspension blocker.";
    this.priorityOutcome = prioritySupported ? "not-attempted" : "unsupported";
    this.priorityDetail = prioritySupported ? "Process priority has not been requested." : "This host does not expose supported owned-process priority controls.";
  }

  onStatus(listener: (event: OsWorkLeaseEvent) => void): () => void {
    this.listeners.add(listener);
    try { listener({ status: this.status() }); }
    catch (error) { this.adapter.diagnostic?.(`OS work status observer failed: ${failure(error)}`); }
    return () => { this.listeners.delete(listener); };
  }

  acquire(value: OsWorkLeaseDescriptor): () => void {
    if (this.closed) throw new Error("OS work lease service is closed.");
    const descriptor = this.checkedDescriptor(value), key = this.key(descriptor.source, descriptor.id), prior = this.leases.get(key);
    if (prior && prior.kind !== descriptor.kind)
      throw new Error("OS work lease identity is already bound to different work.");
    if (prior) { prior.references += 1; prior.label = descriptor.label; }
    else this.leases.set(key, { ...descriptor, references: 1, startedAt: this.now().toISOString() });
    this.activatePlatform(); this.publish();
    let released = false;
    return () => { if (released) return; released = true; this.releaseReference(key); };
  }

  observe(value: OsWorkLeaseDescriptor, active: boolean): void {
    if (this.closed) return;
    const descriptor = this.checkedDescriptor(value), key = this.key(descriptor.source, descriptor.id), prior = this.leases.get(key);
    if (active) {
      if (prior && prior.kind !== descriptor.kind)
        throw new Error("Observed OS work identity changed while it was active.");
      if (!prior) this.leases.set(key, { ...descriptor, references: 1, startedAt: this.now().toISOString() });
      else prior.label = descriptor.label;
      this.activatePlatform(); this.publish(); return;
    }
    if (prior) { this.leases.delete(key); if (!this.leases.size) this.deactivatePlatform(); this.publish(); }
  }

  observeEngineActivity(event: EngineActivityTransition): void {
    this.observeTrusted({ source: "engine", id: event.requestId, kind: event.owner, label: event.label }, ACTIVE_ENGINE_STATES.has(event.state));
  }

  observeRuntimeJob(job: RuntimeJob): void {
    this.observeTrusted({ source: "runtime-job", id: job.id, kind: job.kind, label: job.label }, ACTIVE_JOB_STATES.has(job.state));
  }

  observeStorageOperation(event: BrainStorageOperationEvent): void {
    this.observeTrusted({ source: "storage-operation", id: event.operationId, kind: event.kind, label: event.label },
      ACTIVE_STORAGE_STATES.has(event.state));
  }

  observeActionEvent(event: ActionEvent): void {
    this.observeTrusted({ source: "host", id: event.id, kind: event.action.kind,
      label: event.statusLabel ?? `${event.action.kind} action` }, event.state === "running");
  }

  refreshOwnedProcesses(): void {
    if (!this.closed && this.leases.size) { this.activatePriority(); this.publish(); }
  }

  /** Worker-exit signal: forget the exact dead child without writing to a possibly reused PID. */
  retireOwnedProcess(pid: number): boolean {
    if (!Number.isSafeInteger(pid) || pid <= 0) return false;
    const removed = this.priority.delete(pid);
    if (removed) {
      if (this.leases.size) this.summarizeActivePriority();
      else { this.priorityOutcome = "restored"; this.priorityDetail = "A terminated owned process needed no priority restoration."; }
      this.publish();
    }
    return removed;
  }

  status(): OsWorkLeaseStatus {
    const activeBySource = Object.fromEntries(SOURCES.map(source => [source, 0])) as Record<OsWorkLeaseSource, number>;
    let references = 0;
    for (const lease of this.leases.values()) { activeBySource[lease.source] += 1; references += lease.references; }
    const records = [...this.priority.values()], activeTargets = records.filter(value => value.satisfied).length,
      deniedTargets = records.filter(value => value.denied).length,
      restoreFailedTargets = records.filter(value => value.restoreFailed).length,
      errorTargets = records.filter(value => value.error && !value.denied && !value.restoreFailed).length;
    return { format: "omni-os-work-lease-status", version: 1, platform: this.adapter.platform,
      active: this.leases.size > 0, activeLeaseCount: this.leases.size, activeReferenceCount: references, activeBySource,
      sleepPrevention: { mode: "prevent-app-suspension", active: this.blockerActive(), outcome: this.sleepOutcome, detail: this.sleepDetail },
      processPriority: { requested: "above-normal", ...(this.validRequestedPriority() !== undefined ? { requestedValue: this.validRequestedPriority() } : {}),
        activeTargets, deniedTargets, errorTargets, restoreFailedTargets, outcome: this.priorityOutcome, detail: this.priorityDetail },
      limitations: [...LIMITATIONS], updatedAt: this.now().toISOString() };
  }

  close(): OsWorkLeaseStatus {
    this.closed = true;
    this.leases.clear(); this.deactivatePlatform(); this.publish(); this.listeners.clear(); return this.status();
  }

  private checkedDescriptor(value: OsWorkLeaseDescriptor): OsWorkLeaseDescriptor {
    if (!SOURCES.includes(value?.source)) throw new Error("OS work lease source is invalid.");
    const id = text(value.id, "OS work lease ID", 256); if (!ID.test(id)) throw new Error("OS work lease ID is invalid.");
    return { source: value.source, id, kind: text(value.kind, "OS work kind", 100), label: text(value.label, "OS work label", 200) };
  }
  private observeTrusted(value: OsWorkLeaseDescriptor, active: boolean): void {
    const id = typeof value.id === "string" ? value.id.replace(/\0/g, "").trim() : "";
    if (!ID.test(id)) { this.adapter.diagnostic?.("Ignored an invalid trusted OS-work observation identity."); return; }
    const kind = (typeof value.kind === "string" ? value.kind : "work").replace(/\0/g, "").trim().slice(0, 100) || "work";
    const label = (typeof value.label === "string" ? value.label : `${kind} work`).replace(/\0/g, "").trim().slice(0, 200) || `${kind} work`;
    try { this.observe({ source: value.source, id, kind, label }, active); }
    catch (error) { this.adapter.diagnostic?.(`OS work observation was ignored: ${failure(error)}`); }
  }
  private key(source: OsWorkLeaseSource, id: string): string { return `${source}:${id}`; }
  private releaseReference(key: string): void {
    const lease = this.leases.get(key); if (!lease) return;
    lease.references -= 1; if (lease.references <= 0) this.leases.delete(key);
    if (!this.leases.size) this.deactivatePlatform(); this.publish();
  }
  private publish(): void {
    const event = { status: this.status() };
    for (const listener of this.listeners) try { listener(event); } catch (error) { this.adapter.diagnostic?.(`OS work status observer failed: ${failure(error)}`); }
  }

  private activatePlatform(): void { this.activateSleep(); this.activatePriority(); }
  private deactivatePlatform(): void { this.deactivateSleep(); this.deactivatePriority(); }

  private sleepSupported(): boolean {
    return Boolean(this.adapter.startPowerSaveBlocker && this.adapter.stopPowerSaveBlocker && this.adapter.isPowerSaveBlockerStarted);
  }
  private blockerActive(): boolean {
    if (this.blockerId === undefined || !this.adapter.isPowerSaveBlockerStarted) return false;
    try { return this.adapter.isPowerSaveBlockerStarted(this.blockerId); }
    catch { return false; }
  }
  private activateSleep(): void {
    if (!this.sleepSupported()) return;
    if (this.blockerId !== undefined) {
      try {
        if (this.adapter.isPowerSaveBlockerStarted!(this.blockerId)) { this.sleepOutcome = "active"; return; }
        this.blockerId = undefined;
      } catch (error) {
        this.sleepOutcome = "error";
        this.sleepDetail = `The existing owned blocker could not be inspected; no second blocker was started: ${failure(error)}`;
        return;
      }
    }
    try {
      const id = this.adapter.startPowerSaveBlocker!();
      if (!Number.isSafeInteger(id) || id < 0 || !this.adapter.isPowerSaveBlockerStarted!(id)) {
        if (Number.isSafeInteger(id) && id >= 0) try { this.adapter.stopPowerSaveBlocker!(id); } catch { /* best effort invalid start cleanup */ }
        throw new Error("The host did not confirm its app-suspension blocker.");
      }
      this.blockerId = id; this.sleepOutcome = "active";
      this.sleepDetail = "Best-effort app suspension prevention is active while owned work is running.";
    } catch (error) {
      this.sleepOutcome = denied(error) ? "denied" : "error";
      this.sleepDetail = `App suspension prevention was ${denied(error) ? "denied" : "unavailable"}: ${failure(error)}`;
    }
  }
  private deactivateSleep(): void {
    if (this.blockerId === undefined || !this.adapter.stopPowerSaveBlocker) return;
    const id = this.blockerId;
    try {
      this.adapter.stopPowerSaveBlocker(id);
      if (this.adapter.isPowerSaveBlockerStarted?.(id)) throw new Error("The owned blocker remained active after Stop.");
      this.blockerId = undefined; this.sleepOutcome = "restored";
      this.sleepDetail = "The owned app-suspension blocker was stopped after active work ended.";
    } catch (error) {
      const stillActive = this.adapter.isPowerSaveBlockerStarted ? (() => { try { return this.adapter.isPowerSaveBlockerStarted!(id); } catch { return true; } })() : true;
      if (!stillActive) this.blockerId = undefined;
      this.sleepOutcome = "error";
      this.sleepDetail = `The owned app-suspension blocker could not be confirmed stopped: ${failure(error)}`;
    }
  }

  private validRequestedPriority(): number | undefined {
    const value = this.adapter.requestedPriority;
    return Number.isInteger(value) && Number(value) >= -20 && Number(value) <= 19 ? Number(value) : undefined;
  }
  private prioritySupported(): boolean {
    return Boolean(this.adapter.priorityProcessIds && this.adapter.getPriority && this.adapter.setPriority && this.validRequestedPriority() !== undefined);
  }
  private ownedPids(): number[] {
    if (!this.adapter.priorityProcessIds) return [];
    const values = this.adapter.priorityProcessIds();
    return [...new Set(values.filter(value => Number.isSafeInteger(value) && value > 0))].sort((left, right) => left - right);
  }
  private activatePriority(): void {
    if (!this.prioritySupported()) return;
    let pids: number[];
    try { pids = this.ownedPids(); }
    catch (error) { this.priorityOutcome = "error"; this.priorityDetail = `Owned process discovery failed: ${failure(error)}`; return; }
    for (const pid of pids) {
      const prior = this.priority.get(pid);
      if (prior) { if (prior.applied) prior.restoreFailed = undefined; continue; }
      const record: PriorityRecord = { pid, applied: false, satisfied: false, denied: false }; this.priority.set(pid, record);
      try {
        record.original = this.adapter.getPriority!(pid);
        if (!Number.isInteger(record.original) || record.original < -20 || record.original > 19)
          throw new Error("The host returned an invalid original process priority.");
        if (record.original <= this.validRequestedPriority()!) record.satisfied = true;
        else { this.adapter.setPriority!(pid, this.validRequestedPriority()!); record.applied = true; record.satisfied = true; }
      } catch (error) { record.denied = denied(error); record.error = failure(error); }
    }
    this.summarizeActivePriority();
  }
  private summarizeActivePriority(): void {
    const records = [...this.priority.values()], active = records.filter(value => value.satisfied).length,
      deniedCount = records.filter(value => value.denied).length,
      errorCount = records.filter(value => value.error && !value.denied).length;
    this.priorityOutcome = active && (deniedCount || errorCount) ? "partial" : active ? "active" : deniedCount ? "denied" : errorCount ? "error" : "not-attempted";
    this.priorityDetail = active
      ? `Above-normal priority is active for ${active} owned process${active === 1 ? "" : "es"}${deniedCount || errorCount ? "; other owned targets were unchanged" : ""}.`
      : deniedCount ? "Above-normal priority was denied; owned processes were left unchanged."
        : errorCount ? "Owned process priority could not be changed; see diagnostics." : "No owned process priority target is currently available.";
  }
  private deactivatePriority(): void {
    if (!this.priority.size) return;
    let owned: Set<number>;
    try { owned = new Set(this.ownedPids()); }
    catch (error) {
      for (const record of this.priority.values()) if (record.applied) record.restoreFailed = "Owned process identity could not be revalidated.";
      this.priorityOutcome = "restore-failed";
      this.priorityDetail = `No process priority was restored because current ownership could not be revalidated: ${failure(error)}`;
      return;
    }
    let restored = 0, unchanged = 0, retired = 0, failedRestore = 0, deniedCount = 0, errorCount = 0;
    for (const [pid, record] of [...this.priority.entries()]) {
      if (!owned.has(pid)) { this.priority.delete(pid); retired += 1; continue; }
      if (!record.applied) {
        if (record.satisfied) unchanged += 1;
        deniedCount += record.denied ? 1 : 0; errorCount += record.denied || record.satisfied ? 0 : 1;
        this.priority.delete(pid); continue;
      }
      try {
        this.adapter.setPriority!(pid, record.original!); record.applied = false; record.satisfied = false; record.restoreFailed = undefined;
        this.priority.delete(pid); restored += 1;
      } catch (error) { record.restoreFailed = failure(error); failedRestore += 1; }
    }
    this.priorityOutcome = failedRestore ? "restore-failed" : restored || unchanged ? "restored" : deniedCount ? "denied" : errorCount ? "error" : this.priorityOutcome;
    this.priorityDetail = failedRestore
      ? `${failedRestore} owned process priorit${failedRestore === 1 ? "y was" : "ies were"} not confirmed restored; the next Close or work boundary will retry.`
      : restored ? `Original priority was restored for ${restored} owned process${restored === 1 ? "" : "es"}${retired ? `; ${retired} terminated target${retired === 1 ? " was" : "s were"} forgotten without an OS write` : ""}.`
        : unchanged ? `${unchanged} owned process${unchanged === 1 ? " was" : "es were"} already at equal or higher priority; no change required restoration.`
          : retired ? `${retired} terminated owned process target${retired === 1 ? " was" : "s were"} forgotten without writing to a reusable PID.`
        : deniedCount ? "Above-normal priority was denied; no process priority needed restoration."
          : errorCount ? "Process priority was unchanged after host errors." : this.priorityDetail;
  }
}
