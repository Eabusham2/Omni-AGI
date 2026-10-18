import type { PluginApprovalReview } from "../../shared/plugins";
import type { ActionEvent, ToolExecutionResult } from "../../shared/types";
import "./pluginManager.css";

function plainRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function toolExecutionPluginApprovalReview(
  execution: ToolExecutionResult | undefined,
  identity: { toolId?: string; action?: string }
): PluginApprovalReview | null {
  if (!execution || execution.state !== "approval-required" || !execution.approvalToken ||
    !identity.toolId?.startsWith("plugin.") || execution.toolId !== identity.toolId ||
    execution.action !== identity.action || !plainRecord(execution.output)) return null;
  const record = execution.output as Partial<PluginApprovalReview>;
  if (record.kind !== "plugin-nested-operation" || typeof record.pluginId !== "string" ||
    !/^[a-z0-9][a-z0-9.-]{0,127}$/.test(record.pluginId) ||
    !identity.toolId.startsWith(`plugin.${record.pluginId}.`) ||
    typeof record.pluginName !== "string" || !record.pluginName.trim() || record.pluginName.length > 256 ||
    typeof record.contentSha256 !== "string" || !/^[a-f0-9]{64}$/.test(record.contentSha256) ||
    !record.outer || record.outer.toolId !== identity.toolId || record.outer.action !== identity.action ||
    !record.resolved || typeof record.resolved.toolId !== "string" || record.resolved.toolId.startsWith("plugin.") ||
    typeof record.resolved.action !== "string" || !plainRecord(record.resolved.arguments) ||
    !Array.isArray(record.permissionSources) || record.permissionSources.length !== 2 ||
    !record.permissionSources.every((source) => plainRecord(source) && typeof source.toolId === "string" &&
      ["off", "ask", "auto", "full"].includes(String(source.level)) && typeof source.inherited === "boolean") ||
    !record.permissionSources.some((source) => source.toolId === identity.toolId && source.inherited === false) ||
    !record.permissionSources.some((source) => source.toolId === record.resolved!.toolId && source.inherited === true)) return null;
  return record as PluginApprovalReview;
}

export function pluginApprovalReview(event: ActionEvent): PluginApprovalReview | null {
  if (event.state !== "approval-required") return null;
  return toolExecutionPluginApprovalReview(event.execution, {
    toolId: event.action.toolId,
    action: event.action.action
  });
}

function ResolvedPluginApproval({ review }: { review: PluginApprovalReview }) {
  return <section className="plugin-approval-review" aria-label="Resolved plugin operation">
    <strong>{review.pluginName} resolves to {review.resolved.toolId}.{review.resolved.action}</strong>
    <pre tabIndex={0}>{JSON.stringify(review.resolved.arguments, null, 2)}</pre>
    <small>Package {review.contentSha256.slice(0, 12)}… · both plugin and nested permissions are checked again before execution.</small>
  </section>;
}

/** Mount inside an Ask action card so review names the real nested effect. */
export function PluginApprovalReviewView({ event }: { event: ActionEvent }) {
  const review = pluginApprovalReview(event);
  return review ? <ResolvedPluginApproval review={review} /> : null;
}

/** Manual Tools console variant; uses the real execution without forging an event. */
export function PluginToolApprovalReviewView({ execution, toolId, action }: {
  execution: ToolExecutionResult | undefined;
  toolId: string;
  action: string;
}) {
  const review = toolExecutionPluginApprovalReview(execution, { toolId, action });
  return review ? <ResolvedPluginApproval review={review} /> : null;
}
