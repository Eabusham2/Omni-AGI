import type { ActionEvent } from "../shared/types";
import { McpAppArtifactStore, type McpAppArtifactHooks } from "./mcpAppArtifacts";

const object = (value: unknown): Record<string, unknown> => value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const UUID = /^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i;
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonical((value as Record<string, unknown>)[key])}`).join(",")}}`;
  return JSON.stringify(value);
}
/** Strip only this client's presentation receipt, never arbitrary remote data. */
export function mcpAppOriginalResult(output: unknown): Record<string, unknown> {
  const result = { ...object(output) }, meta = { ...object(result._meta) };
  delete meta["omni/mcpApp"]; delete meta["omni/mcpAppUnavailable"];
  if (Object.keys(meta).length) result._meta = meta; else delete result._meta;
  return result;
}
export async function projectMcpAppAction(brainRoot: string, brainId: string, event: ActionEvent,
  hooks: McpAppArtifactHooks = {}): Promise<ActionEvent> {
  const projected = { ...event }; delete projected.mcpAppView;
  if (event.state !== "complete" || event.execution?.state !== "complete" || !event.action.toolId?.startsWith("mcp.") ||
    event.action.toolId !== event.execution.toolId || event.action.action !== event.execution.action) return projected;
  const hint = object(object(object(event.execution.output)._meta)["omni/mcpApp"]);
  if (typeof hint.id !== "string" || !UUID.test(hint.id)) return projected;
  const store = new McpAppArtifactStore(brainRoot, brainId, hooks), summary = await store.summary(hint.id);
  if (!summary) return projected;
  const expected = canonical(mcpAppOriginalResult(event.execution.output));
  for (const candidate of summary.captures) {
    const loaded = await store.load(hint.id, candidate.sha256);
    if (loaded.capture.tool.id !== hint.toolId || canonical(loaded.result) !== expected) continue;
    // A merge can retain several captures with this artifact ID. Project the
    // actual event's matching provenance, not another currently selected UI.
    const capture = loaded.capture;
    projected.mcpAppView = { id: hint.id, brainId, serverId: capture.source.serverId, serverLabel: capture.source.serverLabel,
      resourceUri: capture.resourceUri, toolId: event.action.toolId, remoteToolName: capture.tool.remoteName, observedAt: capture.observedAt,
      untrusted: true, savedArtifact: true, needsMappingReview: summary.needsMappingReview,
      artifactSha256: candidate.sha256, originalSource: capture.source };
    return projected;
  }
  return projected;
}
