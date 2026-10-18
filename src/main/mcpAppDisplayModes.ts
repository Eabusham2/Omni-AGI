import { randomUUID } from "node:crypto";
import { MCP_APP_DISPLAY_MODES, type McpAppDisplayAcknowledgement, type McpAppDisplayContext,
  type McpAppDisplayEvent, type McpAppDisplayIdentity, type McpAppDisplayMode, type McpAppHostDisplayMode } from "../shared/mcpApps";
import { McpHostRequestError } from "./mcpInteractionProtocol";

interface PendingMode {
  id: string;
  requested: McpAppHostDisplayMode;
  previous: McpAppHostDisplayMode;
  finish(mode: McpAppHostDisplayMode): void;
}
export interface McpAppDisplayLease {
  identity: McpAppDisplayIdentity;
  requireCurrent(): void;
  mode: McpAppHostDisplayMode;
  declared?: McpAppDisplayMode[];
  closed: boolean;
  pending?: PendingMode;
}
const key = (identity: McpAppDisplayIdentity): string => `${identity.brainId}:${identity.viewId}`;
const displayMode = (value: unknown): value is McpAppDisplayMode => ["inline", "fullscreen", "pip"].includes(String(value));

/** The renderer commits presentation before a View receives a successful mode result. */
export class McpAppDisplayModeHost {
  private readonly listeners = new Set<(event: McpAppDisplayEvent) => void>();
  private readonly leases = new Map<string, McpAppDisplayLease>();
  private fullscreen?: McpAppDisplayLease;
  private sequence = 0;
  subscribe(listener: (event: McpAppDisplayEvent) => void): () => void {
    this.listeners.add(listener); return () => this.listeners.delete(listener);
  }
  create(identity: McpAppDisplayIdentity, requireCurrent: () => void): McpAppDisplayLease {
    const previous = this.leases.get(key(identity)); if (previous) this.retire(previous, "The app view was replaced.");
    const lease: McpAppDisplayLease = { identity: { ...identity }, requireCurrent, mode: "inline", closed: false };
    this.leases.set(key(identity), lease); return lease;
  }
  configure(lease: McpAppDisplayLease, capabilities: Record<string, unknown>): void {
    const modes = capabilities.availableDisplayModes;
    if (modes !== undefined && (!Array.isArray(modes) || modes.some(value => typeof value !== "string" || !displayMode(value))))
      throw new McpHostRequestError(-32602, "Invalid MCP App display-mode capabilities.");
    lease.declared = modes === undefined ? undefined : [...new Set(modes as McpAppDisplayMode[])];
  }
  context(lease: McpAppDisplayLease): McpAppDisplayContext {
    return { displayMode: lease.mode, availableDisplayModes: this.listeners.size ? [...MCP_APP_DISPLAY_MODES] : ["inline"] };
  }
  private publish(lease: McpAppDisplayLease, action: McpAppDisplayEvent["action"], extra: Partial<McpAppDisplayEvent> = {}): void {
    const event: McpAppDisplayEvent = { ...lease.identity, ...this.context(lease), sequence: ++this.sequence, action, ...extra };
    for (const listener of this.listeners) { try { listener(structuredClone(event)); } catch { /* a presentation observer grants no authority */ } }
  }
  async request(lease: McpAppDisplayLease, requested: unknown, signal: AbortSignal): Promise<McpAppHostDisplayMode> {
    if (typeof requested !== "string" || !displayMode(requested)) throw new McpHostRequestError(-32602, "Invalid MCP App display mode.");
    signal.throwIfAborted(); lease.requireCurrent();
    if (lease.closed || this.leases.get(key(lease.identity)) !== lease) throw new McpHostRequestError(-32000, "This app presentation expired.");
    if (lease.pending) throw new McpHostRequestError(-32000, "An app display change is already pending.");
    if (requested === lease.mode || requested === "pip" || !this.listeners.size || !lease.declared?.includes(requested)) return lease.mode;
    if (requested === "fullscreen" && this.fullscreen && this.fullscreen !== lease) return lease.mode;
    const previous = lease.mode;
    if (requested === "fullscreen") this.fullscreen = lease;
    return new Promise<McpAppHostDisplayMode>((resolve, reject) => {
      let settled = false;
      const clean = (): void => { clearTimeout(timer); signal.removeEventListener("abort", abort); lease.pending = undefined; };
      const restore = (): void => {
        lease.mode = previous;
        if (previous === "fullscreen") this.fullscreen = lease;
        else if (this.fullscreen === lease) this.fullscreen = undefined;
        if (!lease.closed) this.publish(lease, "context");
      };
      const abort = (): void => { if (settled) return; settled = true; clean(); restore(); reject(signal.reason ?? new Error("App display change stopped.")); };
      const timer = setTimeout(() => { if (settled) return; settled = true; clean(); restore(); resolve(previous); }, 5000);
      lease.pending = { id: randomUUID(), requested, previous, finish: mode => {
        if (settled) return; settled = true; clean(); lease.mode = mode;
        if (mode === "fullscreen") this.fullscreen = lease;
        else if (this.fullscreen === lease) this.fullscreen = undefined;
        this.publish(lease, "context"); resolve(mode);
      } };
      signal.addEventListener("abort", abort, { once: true });
      this.publish(lease, "apply", { operationId: lease.pending.id, displayMode: requested });
    });
  }
  acknowledge(input: McpAppDisplayAcknowledgement): void {
    if (!input || Object.keys(input).some(field => !["brainId", "viewId", "viewToken", "operationId", "mode"].includes(field)) ||
        typeof input.brainId !== "string" || typeof input.viewId !== "string" || typeof input.viewToken !== "string" ||
        typeof input.operationId !== "string" || !["inline", "fullscreen"].includes(input.mode))
      throw new Error("Invalid app display acknowledgement.");
    const lease = this.leases.get(key(input));
    if (!lease || lease.closed || lease.identity.viewToken !== input.viewToken || lease.pending?.id !== input.operationId)
      throw new Error("This app display acknowledgement expired.");
    lease.requireCurrent();
    if (input.mode !== lease.pending.requested && input.mode !== lease.pending.previous)
      throw new Error("The app renderer acknowledged an unrequested display mode.");
    lease.pending.finish(input.mode);
  }
  retire(lease: McpAppDisplayLease, reason: string): void {
    if (lease.closed) return;
    lease.closed = true; lease.pending?.finish(lease.mode);
    if (this.fullscreen === lease) this.fullscreen = undefined;
    if (this.leases.get(key(lease.identity)) === lease) this.leases.delete(key(lease.identity));
    this.publish(lease, "closed", { displayMode: "inline", reason });
  }
  dispose(): void {
    for (const lease of this.leases.values()) this.retire(lease, "The connected-app host closed.");
    this.listeners.clear();
  }
}
