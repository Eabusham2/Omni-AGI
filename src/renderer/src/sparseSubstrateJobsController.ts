import type { SparseSubstrateJobView, SparseSubstrateUiApi } from "../../shared/sparseSubstrateTransfer";

export interface SparseJobsSnapshot {
  brainId: string;
  jobs: SparseSubstrateJobView[];
  loading: boolean;
  paging: boolean;
  nextAfter?: string;
  error?: string;
  jobErrors: Record<string, string>;
  busyJobIds: string[];
}
export const emptySparseJobs = (brainId: string): SparseJobsSnapshot => ({ brainId, jobs: [], loading: false, paging: false, jobErrors: {}, busyJobIds: [] });
const message = (error: unknown) => error instanceof Error ? error.message : String(error);

/** One live subscription per mounted identity. Reads never grant neural work. */
export class SparseJobsController {
  private active = true;
  private unsubscribe?: () => void;
  private jobs = new Map<string, SparseSubstrateJobView>();
  private reads = new Map<string, Promise<void>>();
  private stopping = new Map<string, Promise<void>>();
  private page?: Promise<void>;
  private snapshot: SparseJobsSnapshot;
  constructor(readonly brainId: string, private api: SparseSubstrateUiApi, private changed: (snapshot: SparseJobsSnapshot) => void) {
    this.snapshot = emptySparseJobs(brainId);
  }
  private emit(): void {
    if (!this.active) return;
    this.snapshot = { ...this.snapshot, jobs: [...this.jobs.values()].sort((a, b) => b.createdAt.localeCompare(a.createdAt) || a.id.localeCompare(b.id)),
      jobErrors: { ...this.snapshot.jobErrors }, busyJobIds: [...this.stopping.keys()] };
    try { this.changed(this.snapshot); } catch { /* A UI observer cannot own a job's outcome. */ }
  }
  private merge(job: SparseSubstrateJobView): void {
    if (!this.active || job.brainId !== this.brainId || !Number.isSafeInteger(job.revision) || job.revision < 1) return;
    const old = this.jobs.get(job.id);
    if (!old || old.revision < job.revision) this.jobs.set(job.id, structuredClone(job));
  }
  start(): void {
    if (!this.active || this.unsubscribe) return;
    this.unsubscribe = this.api.onEvent(this.brainId, event => {
      if (!this.active || event.brainId !== this.brainId) return;
      if (event.kind === "job") this.merge(event.job);
      else this.snapshot.error = event.error;
      this.emit();
    });
    void this.refresh();
  }
  stop(): void { this.active = false; this.unsubscribe?.(); this.unsubscribe = undefined; }
  refresh(): Promise<void> { return this.loadPage(); }
  older(): Promise<void> { return this.snapshot.nextAfter ? this.loadPage(this.snapshot.nextAfter) : Promise.resolve(); }
  private loadPage(after?: string): Promise<void> {
    if (!this.active) return Promise.resolve();
    if (this.page) return this.page;
    this.snapshot = { ...this.snapshot, loading: !after, paging: Boolean(after), error: undefined }; this.emit();
    const operation = (async () => {
      try {
        const result = await this.api.jobs(this.brainId, { ...(after ? { after } : {}), limit: 40 });
        if (!this.active) return;
        result.entries.forEach(job => this.merge(job)); this.snapshot.nextAfter = result.nextAfter;
      } catch (error) { if (this.active) this.snapshot.error = message(error); }
      finally { if (this.active) { this.snapshot.loading = false; this.snapshot.paging = false; this.emit(); } }
    })();
    this.page = operation;
    void operation.finally(() => { if (this.page === operation) this.page = undefined; });
    return operation;
  }
  ensure(jobId: string): Promise<void> {
    if (this.jobs.has(jobId)) return Promise.resolve();
    return this.refreshJob(jobId);
  }
  refreshJob(jobId: string): Promise<void> {
    if (!this.active) return Promise.resolve();
    const old = this.reads.get(jobId); if (old) return old;
    const operation = (async () => {
      try { const job = await this.api.job(this.brainId, jobId); if (!this.active) return;
        if (job.id !== jobId || job.brainId !== this.brainId) throw new Error("This update belongs to another instance.");
        this.merge(job); delete this.snapshot.jobErrors[jobId];
      } catch (error) { if (this.active) this.snapshot.jobErrors[jobId] = message(error); }
      finally { this.reads.delete(jobId); this.emit(); }
    })();
    this.reads.set(jobId, operation); return operation;
  }
  stopJob(jobId: string): Promise<void> {
    if (!this.active) return Promise.resolve();
    const old = this.stopping.get(jobId); if (old) return old;
    const operation = (async () => {
      try {
        const result = await this.api.stop(this.brainId, jobId);
        if (!this.active) return;
        if (result.id !== jobId || result.brainId !== this.brainId) throw new Error("Stop returned a different update.");
        this.merge(result); delete this.snapshot.jobErrors[jobId];
      } catch (error) { if (this.active) this.snapshot.jobErrors[jobId] = message(error); }
      finally { this.stopping.delete(jobId); this.emit(); }
    })();
    this.stopping.set(jobId, operation); this.emit(); return operation;
  }
}
