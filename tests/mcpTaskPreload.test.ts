import { expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";
const bridge = vi.hoisted(() => ({ invoke: vi.fn(async () => undefined), expose: vi.fn() }));
vi.mock("electron", () => ({ contextBridge: { exposeInMainWorld: bridge.expose },
  ipcRenderer: { invoke: bridge.invoke, on: vi.fn(), removeListener: vi.fn() }, webUtils: { getPathForFile: vi.fn() } }));
it("exposes persisted local MCP task pages without a renderer dispatch/rebind credential capability", async () => {
  await import("../src/preload/index");
  const api = bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi;
  await api.mcp.tasks!.localPage("brain-a", "opaque-current-page", 16);
  expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.mcp.tasksLocalPage, "brain-a", "opaque-current-page", 16);
  await api.mcp.tasks!.stopWait("brain-a", "owned-task-ref");
  expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.mcp.tasksStopWait, "brain-a", "owned-task-ref");
  expect(api.mcp.tasks).not.toHaveProperty("start"); expect(api.mcp.tasks).not.toHaveProperty("rebind");
  expect(api.mcp.tasks).not.toHaveProperty("getCredential");
});
