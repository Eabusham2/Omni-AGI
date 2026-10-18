import { expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";
const bridge = vi.hoisted(() => ({ invoke: vi.fn(async () => undefined), expose: vi.fn() }));
vi.mock("electron", () => ({ contextBridge: { exposeInMainWorld: bridge.expose },
  ipcRenderer: { invoke: bridge.invoke, on: vi.fn(), removeListener: vi.fn() }, webUtils: { getPathForFile: vi.fn() } }));
it("bridges only reviewed native operation intent and bounded plan metadata", async () => {
  await import("../src/preload/index");
  const api = bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi;
  expect(Object.keys(api.nativeOperationAuto)).toEqual(["prepare", "arm", "status", "plans", "stop"]);
  await api.nativeOperationAuto.prepare({ brainId: "a" });
  expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.nativeOperationAuto.prepare, { brainId: "a" });
  await api.nativeOperationAuto.plans({ brainId: "a", limit: 8, after: "opaque" });
  expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.nativeOperationAuto.plans, { brainId: "a", limit: 8, after: "opaque" });
  await api.nativeOperationAuto.stop({ brainId: "a", operationId: "exact" });
  expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.nativeOperationAuto.stop, { brainId: "a", operationId: "exact" });
  expect(api.nativeOperationAuto).not.toHaveProperty("load"); expect(api.nativeOperationAuto).not.toHaveProperty("calibrate");
});
