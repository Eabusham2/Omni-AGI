import type { PluginRuntimeApi, PluginRuntimePendingApproval } from "../../shared/pluginRuntime";

export interface RuntimeApprovalSnapshot {
  brainId?: string;
  approvals: PluginRuntimePendingApproval[];
  busyReviewIds: string[];
  error: string;
  message: string;
}
export const EMPTY_RUNTIME_APPROVALS: RuntimeApprovalSnapshot = { approvals: [], busyReviewIds: [], error: "", message: "" };
export function ownedRuntimeApprovalSnapshot(brainId: string, snapshot: RuntimeApprovalSnapshot): RuntimeApprovalSnapshot {
  return snapshot.brainId === brainId ? snapshot : EMPTY_RUNTIME_APPROVALS;
}

/** One bounded in-flight read per owner; a stopped view never publishes again. */
export class RuntimeApprovalController {
  private stopped = false;
  private reading = false;
  private dirty = false;
  private unsubscribe?: () => void;
  private actionError = "";
  private readonly busy = new Set<string>();
  private snapshot: RuntimeApprovalSnapshot;
  constructor(private readonly api: PluginRuntimeApi, readonly brainId: string,
    private readonly changed: (snapshot: RuntimeApprovalSnapshot) => void) {
    this.snapshot = { ...EMPTY_RUNTIME_APPROVALS, brainId };
  }
  start(): void {
    if (this.stopped || this.unsubscribe) return;
    this.unsubscribe = this.api.onEvent(event => { if (event.brainId === this.brainId) void this.refresh(); });
    void this.refresh();
  }
  stop(): void { this.stopped = true; this.dirty = false; this.unsubscribe?.(); this.unsubscribe = undefined; }
  private publish(): void { if (!this.stopped) this.changed({ ...this.snapshot, approvals: [...this.snapshot.approvals], busyReviewIds: [...this.busy] }); }
  async refresh(): Promise<void> {
    if (this.stopped) return;
    if (this.reading) { this.dirty = true; return; }
    this.reading = true;
    try {
      const approvals = await this.api.approvals(this.brainId);
      if (this.stopped) return;
      this.snapshot = { ...this.snapshot, approvals: approvals.filter(item => item.brainId === this.brainId), error: this.actionError };
      this.publish();
    } catch (error) {
      if (!this.stopped) { this.snapshot = { ...this.snapshot, error: error instanceof Error ? error.message : String(error) }; this.publish(); }
    } finally {
      this.reading = false;
      if (!this.stopped && this.dirty) { this.dirty = false; void this.refresh(); }
    }
  }
  async review(item: PluginRuntimePendingApproval, decision: "approve" | "deny"): Promise<void> {
    if (this.stopped || item.brainId !== this.brainId || this.busy.has(item.reviewId) ||
        !this.snapshot.approvals.some(current => current.sessionId === item.sessionId && current.rpcId === item.rpcId && current.reviewId === item.reviewId && current.contentSha256 === item.contentSha256)) return;
    this.busy.add(item.reviewId); this.actionError = ""; this.snapshot = { ...this.snapshot, error: "", message: "" }; this.publish();
    try {
      const result = await this.api.review({ brainId: this.brainId, sessionId: item.sessionId, rpcId: item.rpcId, reviewId: item.reviewId, decision });
      if (decision === "approve" && (result.state === "failed" || result.state === "uncertain")) throw new Error(result.error ?? "The reviewed action did not complete; no automatic retry was made.");
      if (!this.stopped) this.snapshot = { ...this.snapshot, message: result.state === "approval-required"
        ? "The action needs a new exact review."
        : decision === "approve" ? "Exact action approved." : "Exact action denied." };
    } catch (error) {
      if (!this.stopped) { this.actionError = error instanceof Error ? error.message : String(error); this.snapshot = { ...this.snapshot, error: this.actionError }; }
    } finally {
      this.busy.delete(item.reviewId);
      if (!this.stopped) { this.publish(); await this.refresh(); }
    }
  }
}
