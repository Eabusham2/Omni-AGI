import { expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";
const bridge = vi.hoisted(() => ({ invoke: vi.fn(async () => undefined), expose: vi.fn() }));
vi.mock("electron", () => ({ contextBridge: { exposeInMainWorld: bridge.expose },
  ipcRenderer: { invoke: bridge.invoke, on: vi.fn(), removeListener: vi.fn() }, webUtils: { getPathForFile: vi.fn() } }));
it("exposes best-effort work status read-only, never renderer-selected process priority or PIDs", async () => {
  await import("../src/preload/index");
  const api = bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi;
  await api.window.workStatus(); expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.window.workStatus);
  expect(api.window).not.toHaveProperty("setPriority"); expect(api.window).not.toHaveProperty("setPowerSaveBlocker");
});
