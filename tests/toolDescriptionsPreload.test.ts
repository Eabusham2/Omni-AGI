import { describe, expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";
const bridge = vi.hoisted(() => ({ expose: vi.fn(), invoke: vi.fn(), on: vi.fn(), off: vi.fn() }));
vi.mock("electron", () => ({ contextBridge: { exposeInMainWorld: bridge.expose }, ipcRenderer: { invoke: bridge.invoke, on: bridge.on, removeListener: bridge.off }, webUtils: { getPathForFile: vi.fn() } }));
import "../src/preload/index";
describe("read-only tool description preload", () => {
  it("sends only the current owner ID and never an execution or grant", async () => {
    const api = bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi;
    await api.tool.descriptions("brain-a"); expect(bridge.invoke).toHaveBeenCalledWith(IPC.tool.descriptions, "brain-a");
    expect(bridge.invoke).not.toHaveBeenCalledWith(IPC.tool.execute, expect.anything());
  });
});
