import { expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";
const bridge = vi.hoisted(() => ({ invoke: vi.fn(async () => undefined), expose: vi.fn(), on: vi.fn(), off: vi.fn() }));
vi.mock("electron", () => ({ contextBridge: { exposeInMainWorld: bridge.expose }, ipcRenderer: { invoke: bridge.invoke, on: bridge.on, removeListener: bridge.off }, webUtils: { getPathForFile: vi.fn() } }));
it("scopes presentation acknowledgements to their ephemeral view and exposes a removable read-only display listener", async () => {
  await import("../src/preload/index"); const api = bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi;
  const ack = { brainId: "brain", viewId: "view", viewToken: "a".repeat(64), operationId: "owned-presentation", mode: "inline" as const };
  await api.mcp.apps.acknowledgeDisplayMode!(ack); expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.mcp.appDisplayModeAcknowledge, ack);
  const stop = api.mcp.apps.onDisplayMode!(() => undefined); expect(bridge.on.mock.calls.at(-1)?.[0]).toBe(IPC.mcp.appDisplayMode);
  stop(); expect(bridge.off.mock.calls.at(-1)?.[0]).toBe(IPC.mcp.appDisplayMode);
  expect(api.mcp.apps).not.toHaveProperty("openWindow"); expect(api.mcp.apps).not.toHaveProperty("grantDevices");
});
