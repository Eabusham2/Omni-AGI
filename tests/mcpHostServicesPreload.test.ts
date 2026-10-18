import { expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";
const bridge = vi.hoisted(() => ({ invoke: vi.fn(async () => undefined), expose: vi.fn(), on: vi.fn(), off: vi.fn() }));
vi.mock("electron", () => ({ contextBridge: { exposeInMainWorld: bridge.expose }, ipcRenderer: { invoke: bridge.invoke, on: bridge.on, removeListener: bridge.off }, webUtils: { getPathForFile: vi.fn() } }));
it("exposes disclosure intent without paths and scopes App notifications to a removable listener", async () => {
  await import("../src/preload/index"); const api = bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi;
  await api.mcp.roots.select({ brainId: "a", serverId: "s", requestId: "r" });
  expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.mcp.rootsSelect, { brainId: "a", serverId: "s", requestId: "r" });
  await api.mcp.diagnostics.page("a", "s", 10, 20); expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.mcp.diagnosticsPage, "a", "s", 10, 20);
  const stop = api.mcp.apps.onServerNotification!(() => undefined); expect(bridge.on.mock.calls.at(-1)?.[0]).toBe(IPC.mcp.appServerNotification);
  stop(); expect(bridge.off.mock.calls.at(-1)?.[0]).toBe(IPC.mcp.appServerNotification);
});
