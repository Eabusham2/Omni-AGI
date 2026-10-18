import type { McpAppDisplayEvent, McpAppDisplayIdentity, McpAppHostDisplayMode } from "../../shared/mcpApps";

/** Reject delayed events from a closed/replaced iframe before touching its presentation. */
export class McpAppDisplayGate {
  private sequence = -1;
  private closed = false;
  constructor(private readonly identity: McpAppDisplayIdentity) {}
  close(): void { this.closed = true; }
  current(sequence: number): boolean { return !this.closed && this.sequence === sequence; }
  receive(event: McpAppDisplayEvent): McpAppDisplayEvent | undefined {
    if (this.closed || !event || event.brainId !== this.identity.brainId || event.viewId !== this.identity.viewId || event.viewToken !== this.identity.viewToken ||
        !Number.isSafeInteger(event.sequence) || event.sequence <= this.sequence || !["apply", "context", "closed"].includes(event.action) ||
        !["inline", "fullscreen"].includes(event.displayMode) || !Array.isArray(event.availableDisplayModes) ||
        event.availableDisplayModes.some(mode => !["inline", "fullscreen"].includes(mode)) ||
        event.action === "apply" && (typeof event.operationId !== "string" || !event.operationId)) return undefined;
    this.sequence = event.sequence; if (event.action === "closed") this.closed = true; return event;
  }
}
/** A top-layer dialog changes presentation without moving or remounting the iframe. */
export function presentMcpAppDisplayMode(surface: HTMLDialogElement | null, requested: McpAppHostDisplayMode): McpAppHostDisplayMode {
  if (!surface?.isConnected) return "inline";
  const current = (): McpAppHostDisplayMode => surface.matches(":modal") ? "fullscreen" : "inline";
  try {
    if (current() === requested) return requested;
    if (surface.open) surface.close();
    if (requested === "fullscreen") surface.showModal(); else surface.show();
    return current();
  } catch {
    // A refused browser presentation never becomes a successful fullscreen ACK.
    if (!surface.open) { try { surface.show(); } catch { /* closing view may have disconnected */ } }
    return current();
  }
}
