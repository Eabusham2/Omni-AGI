import { useEffect, useRef, useState } from "react";
import type { PluginRuntimeApi, PluginRuntimePendingApproval } from "../../shared/pluginRuntime";
import { EMPTY_RUNTIME_APPROVALS, RuntimeApprovalController, ownedRuntimeApprovalSnapshot, type RuntimeApprovalSnapshot } from "./runtimeApprovalController";

export function PluginRuntimeApprovalCard({ item, busy, onReview }: {
  item: PluginRuntimePendingApproval;
  busy: boolean;
  onReview(item: PluginRuntimePendingApproval, decision: "approve" | "deny"): void;
}) {
  return <section className="plugin-runtime-approval">
    <strong>Host action needs review</strong>
    <p>Review the exact host operation before it runs.</p>
    <pre>{JSON.stringify({ toolId: item.invocation.toolId, action: item.invocation.action, arguments: item.invocation.arguments }, null, 2)}</pre>
    <details><summary>Request details</summary><dl>
      <div><dt>Execution ID</dt><dd>{item.executionId}</dd></div>
      <div><dt>Request ID</dt><dd>{item.rpcId}</dd></div>
      <div><dt>Module</dt><dd>{item.pluginId} · {item.moduleId}</dd></div>
    </dl></details>
    <div className="plugin-manager__actions">
      <button type="button" className="button" disabled={busy} onClick={() => onReview(item, "deny")}>Deny</button>
      <button type="button" className="button button--primary" disabled={busy} onClick={() => onReview(item, "approve")}>Approve exact action</button>
    </div>
  </section>;
}

/** Available during initial training, before chat or a model is loaded. */
export function PluginRuntimeApprovalQueue({ brainId, api = typeof window === "undefined" ? undefined : window.omni?.pluginRuntime }: {
  brainId: string;
  api?: PluginRuntimeApi;
}) {
  const [snapshot, setSnapshot] = useState<RuntimeApprovalSnapshot>(EMPTY_RUNTIME_APPROVALS);
  const controller = useRef<RuntimeApprovalController | undefined>(undefined);
  const current = ownedRuntimeApprovalSnapshot(brainId, snapshot);
  useEffect(() => {
    if (!api) return;
    const owned = new RuntimeApprovalController(api, brainId, setSnapshot);
    controller.current = owned; owned.start();
    return () => { owned.stop(); if (controller.current === owned) controller.current = undefined; };
  }, [api, brainId]);
  if (!current.approvals.length && !current.error && !current.message) return null;
  return <section className="plugin-runtime-training-approvals" aria-label="Training action review">
    {current.approvals.length ? <p role="status">{current.approvals.some(item => !current.busyReviewIds.includes(item.reviewId))
      ? "Training is waiting for action review." : "Reviewed actions are running."}</p> : null}
    {current.approvals.map(item => <PluginRuntimeApprovalCard key={`${item.sessionId}/${item.rpcId}/${item.reviewId}`} item={item}
      busy={current.busyReviewIds.includes(item.reviewId)} onReview={(review, decision) => {
        if (controller.current?.brainId === brainId) void controller.current.review(review, decision);
      }} />)}
    {current.message ? <p role="status">{current.message}</p> : null}
    {current.error ? <div role="alert"><p>{current.error}</p><button type="button" className="button" onClick={() => {
      if (controller.current?.brainId === brainId) void controller.current.refresh();
    }}>Refresh action review</button></div> : null}
  </section>;
}
