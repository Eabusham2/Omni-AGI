import { expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";

const host = vi.hoisted(() => ({ invoke: vi.fn(async () => undefined), expose: vi.fn() }));
vi.mock("electron", () => ({ contextBridge: { exposeInMainWorld: host.expose },
  ipcRenderer: { invoke: host.invoke, on: vi.fn(), removeListener: vi.fn() }, webUtils: { getPathForFile: vi.fn() } }));

it("exposes only typed regional review and epoch operations, never worker paths/topology execution", async () => {
  await import("../src/preload/index");
  const api = host.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi;
  expect(Object.keys(api.regionalExecution)).toEqual(["prepare", "status", "activate", "deactivate"]);
  await api.regionalExecution.prepare({ brainId: "fixture", start: 256, count: 256 });
  expect(host.invoke).toHaveBeenLastCalledWith(IPC.regionalExecution.prepare, { brainId: "fixture", start: 256, count: 256 });
  await api.regionalExecution.status("fixture"); expect(host.invoke).toHaveBeenLastCalledWith(IPC.regionalExecution.status, "fixture");
  await api.regionalExecution.activate({ brainId: "fixture", reviewId: "main-owned-token" });
  expect(host.invoke).toHaveBeenLastCalledWith(IPC.regionalExecution.activate, { brainId: "fixture", reviewId: "main-owned-token" });
  await api.regionalExecution.deactivate({ brainId: "fixture", epoch: "existing-epoch" });
  expect(host.invoke).toHaveBeenLastCalledWith(IPC.regionalExecution.deactivate, { brainId: "fixture", epoch: "existing-epoch" });
  expect(api.regionalExecution).not.toHaveProperty("execute"); expect(api.regionalExecution).not.toHaveProperty("load");
});
