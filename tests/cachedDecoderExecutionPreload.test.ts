import { expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";

const bridge = vi.hoisted(() => ({ invoke: vi.fn(async () => undefined), expose: vi.fn() }));
vi.mock("electron", () => ({ contextBridge: { exposeInMainWorld: bridge.expose },
  ipcRenderer: { invoke: bridge.invoke, on: vi.fn(), removeListener: vi.fn() }, webUtils: { getPathForFile: vi.fn() } }));

it("exposes cached comparison intent without renderer-owned topology, load or parameter authority", async () => {
  await import("../src/preload/index");
  const api = bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi;
  expect(Object.keys(api.cachedDecoderExecution)).toEqual(["prepare", "arm", "status", "stop"]);
  await api.cachedDecoderExecution.prepare({ brainId: "fixture" });
  expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.cachedDecoderExecution.prepare, { brainId: "fixture" });
  await api.cachedDecoderExecution.arm({ brainId: "fixture", reviewId: "review" });
  expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.cachedDecoderExecution.arm, { brainId: "fixture", reviewId: "review" });
  await api.cachedDecoderExecution.stop({ brainId: "fixture", operationId: "operation" });
  expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.cachedDecoderExecution.stop, { brainId: "fixture", operationId: "operation" });
  expect(api.cachedDecoderExecution).not.toHaveProperty("load"); expect(api.cachedDecoderExecution).not.toHaveProperty("paths");
});
