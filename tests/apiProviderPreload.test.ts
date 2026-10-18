import { expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";
const bridge = vi.hoisted(() => ({ invoke: vi.fn(async () => undefined), expose: vi.fn() }));
vi.mock("electron", () => ({ contextBridge: { exposeInMainWorld: bridge.expose },
  ipcRenderer: { invoke: bridge.invoke, on: vi.fn(), removeListener: vi.fn() }, webUtils: { getPathForFile: vi.fn() } }));
it("routes optional provider metadata and exact durable job controls through preload, not renderer filesystem access", async () => {
  await import("../src/preload/index");
  const api = bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi;
  const input = { channel: "reviewer" as const, requestId: "discovery-a", expectedSettingsRevision: 3, expectedCredentialRevision: 2 };
  await api.advice.discoverModels("brain-a", input);
  expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.advice.discoverModels, "brain-a", input);
  await api.advice.cancelModelDiscovery("brain-a", { channel: "reviewer", requestId: "discovery-a" });
  expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.advice.cancelModelDiscovery, "brain-a", { channel: "reviewer", requestId: "discovery-a" });
  for (const method of ["pause", "resume", "stop", "details"] as const) {
    await api.teacher[method]({ brainId: "brain-a", jobId: "job-a" });
    expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.teacher[method], { brainId: "brain-a", jobId: "job-a" });
  }
  const resolution = { brainId: "brain-a", jobId: "job-a", itemIndex: 2, decision: "skip" as const };
  await api.teacher.resolveUncertain(resolution);
  expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.teacher.resolveUncertain, resolution);
  expect(api.teacher).not.toHaveProperty("readFile"); expect(api.teacher).not.toHaveProperty("getCredential");
});
