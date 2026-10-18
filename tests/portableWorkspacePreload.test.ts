import { expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";

const host = vi.hoisted(() => ({ invoke: vi.fn(async () => undefined), expose: vi.fn() }));
vi.mock("electron", () => ({
  contextBridge: { exposeInMainWorld: host.expose },
  ipcRenderer: { invoke: host.invoke, on: vi.fn(), removeListener: vi.fn() },
  webUtils: { getPathForFile: vi.fn() }
}));

it("exposes typed portable intent and review only, retaining existing storage cancellation", async () => {
  await import("../src/preload/index");
  const api = host.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi;
  expect(Object.keys(api.portable)).toEqual(["create", "prepare", "status", "reviewDestination", "approveDestination"]);
  const request = { brainId: "selected-instance", requestId: "portable-copy", presentation: {
    appearance: { schemaVersion: 1 as const, mode: "system" as const, palette: "aqua" as const, layout: "glass" as const },
    workspace: { schemaVersion: 1 as const, arrangement: "split" as const, notificationsEnabled: false, profiles: [] }, theme: null
  } };
  await api.portable.create(request); expect(host.invoke).toHaveBeenLastCalledWith(IPC.portable.create, request);
  await api.portable.prepare({ requestId: "destination-copy" }); expect(host.invoke).toHaveBeenLastCalledWith(IPC.portable.prepare, { requestId: "destination-copy" });
  await api.portable.status(); expect(host.invoke).toHaveBeenLastCalledWith(IPC.portable.status);
  await api.portable.reviewDestination(); expect(host.invoke).toHaveBeenLastCalledWith(IPC.portable.reviewDestination);
  const review = { reviewId: "exact-review", reviewSha256: "a".repeat(64) };
  await api.portable.approveDestination(review); expect(host.invoke).toHaveBeenLastCalledWith(IPC.portable.approveDestination, review);
  await api.brain.cancelStorageOperation(request.requestId);
  expect(host.invoke).toHaveBeenLastCalledWith(IPC.brain.cancelStorageOperation, request.requestId);
  expect(api.portable).not.toHaveProperty("execute"); expect(api.portable).not.toHaveProperty("paths");
});
