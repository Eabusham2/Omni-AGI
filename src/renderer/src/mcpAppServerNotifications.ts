import type { McpAppLaunch } from "../../shared/mcpApps";
import type { McpAppServerNotification } from "../../shared/mcpHostServices";

/** One live iframe lease. It forwards invalidation, never a tool or refresh. */
export class McpAppNotificationGate {
  private sequence = -1;
  private epoch?: number;
  private closed = false;
  constructor(private readonly launch: McpAppLaunch) {}
  close(): void { this.closed = true; }
  receive(event: McpAppServerNotification, ready: boolean): { jsonrpc: "2.0"; method: string; params: Record<string, never> } | undefined {
    if (this.closed || !ready || event.brainId !== this.launch.view.brainId || event.serverId !== this.launch.view.serverId ||
        event.viewId !== this.launch.view.id || event.viewToken !== this.launch.viewToken ||
        !Number.isSafeInteger(event.connectionEpoch) || event.connectionEpoch < 0 || !Number.isSafeInteger(event.sequence) || event.sequence <= this.sequence ||
        !["notifications/tools/list_changed", "notifications/resources/list_changed"].includes(event.method) || !event.params ||
        typeof event.params !== "object" || Array.isArray(event.params) || Object.keys(event.params).length || this.epoch !== undefined && this.epoch !== event.connectionEpoch)
      return undefined;
    this.sequence = event.sequence; this.epoch = event.connectionEpoch;
    return { jsonrpc: "2.0", method: event.method, params: {} };
  }
}
