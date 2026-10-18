import { createHash } from "node:crypto";
import type { BrainService } from "./brainService";
import type { McpClientService } from "./mcpClient";

function connectionId(serverId: string): string { return `mcp:connection:${serverId}`; }
function subscriptionId(serverId: string, uri: string): string {
  return `mcp:subscription:${serverId}:${createHash("sha256").update(uri).digest("hex")}`;
}
export function suspendMcpForNativeOwnerRelease(service: Pick<BrainService, "instanceLifecycle">,
  mcp: Pick<McpClientService, "suspendBrain">, brainId: string): void {
  const receipt = mcp.suspendBrain(brainId);
  for (const serverId of receipt.disconnectedServerIds) service.instanceLifecycle.rememberReconnect(brainId, {
    id: connectionId(serverId), kind: "remote-session", label: `MCP ${serverId}`,
    reason: "Native RAM ownership ended. Explicit Load and a new permitted connection are required; completed effects are not replayed.", permissionRequired: true
  });
  for (const entry of receipt.lostResourceSubscriptions) service.instanceLifecycle.rememberReconnect(brainId, {
    id: subscriptionId(entry.serverId, entry.uri), kind: "remote-session", label: `MCP resource ${entry.uri}`,
    reason: "Native RAM ownership ended. Request this resource subscription again explicitly; it was not replayed.", permissionRequired: true
  });
}

/** Main-owned hook: the native acknowledged Unload, not an initial unloaded card. */
export function installMcpUnloadBoundary(
  service: Pick<BrainService, "addInstanceUnloadBoundary" | "instanceLifecycle">,
  mcp: Pick<McpClientService, "suspendBrain" | "onCapabilityEvent">
): () => void {
  const removeBoundary = service.addInstanceUnloadBoundary(brainId => {
    const receipt = mcp.suspendBrain(brainId);
    for (const serverId of receipt.disconnectedServerIds) service.instanceLifecycle.rememberReconnect(brainId, {
      id: connectionId(serverId), kind: "remote-session", label: `MCP ${serverId}`,
      reason: "The external connection closed on Unload. A new permitted MCP action establishes a fresh session; completed effects are not replayed.",
      permissionRequired: true
    });
    for (const entry of receipt.lostResourceSubscriptions) service.instanceLifecycle.rememberReconnect(brainId, {
      id: subscriptionId(entry.serverId, entry.uri), kind: "remote-session",
      label: `MCP resource ${entry.uri.length > 200 ? `${entry.uri.slice(0, 197)}...` : entry.uri}`,
      reason: "Request this subscription again explicitly. Unload does not preserve or replay external subscriptions.",
      permissionRequired: true
    });
  });
  const removeObserver = mcp.onCapabilityEvent(event => {
    if (event.kind === "connection-ready") service.instanceLifecycle.acknowledgeReconnect(event.brainId, [connectionId(event.serverId)]);
    if ((event.kind === "resource-subscribed" || event.kind === "resource-unsubscribed") && event.uri) {
      service.instanceLifecycle.acknowledgeReconnect(event.brainId, [subscriptionId(event.serverId, event.uri)]);
    }
  });
  return () => { removeBoundary(); removeObserver(); };
}
