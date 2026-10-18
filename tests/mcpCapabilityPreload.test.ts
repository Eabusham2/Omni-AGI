import { describe, expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";
import type { McpCapabilityEvent } from "../src/shared/mcpCapabilities";

const bridge = vi.hoisted(() => ({ expose: vi.fn(), on: vi.fn(), off: vi.fn(), invoke: vi.fn() }));
vi.mock("electron", () => ({
  contextBridge: { exposeInMainWorld: bridge.expose },
  ipcRenderer: { on: bridge.on, removeListener: bridge.off, invoke: bridge.invoke },
  webUtils: { getPathForFile: vi.fn() }
}));
import "../src/preload/index";

describe("read-only MCP capability observations", () => {
  it("exposes removable typed observation without creating an RPC dispatch bypass", () => {
    const api = bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi;
    const listener = vi.fn(); const remove = api.mcp.onCapabilityEvent(listener);
    const [channel, wrapped] = bridge.on.mock.calls.at(-1)!;
    expect(channel).toBe(IPC.mcp.capabilityEvent);
    const event: McpCapabilityEvent = { brainId: "first", serverId: "library", revision: 1,
      observedAt: new Date(0).toISOString(), kind: "resource-updated", uri: "fixture://document" };
    wrapped({}, event); expect(listener).toHaveBeenCalledWith(event);
    expect(bridge.invoke).not.toHaveBeenCalled();
    expect("executeOperation" in api.mcp).toBe(false);
    remove(); expect(bridge.off).toHaveBeenCalledWith(channel, wrapped);
  });
});
