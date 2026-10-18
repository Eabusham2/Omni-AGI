import { expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";

const host = vi.hoisted(() => ({ invoke: vi.fn(async () => undefined), expose: vi.fn() }));
vi.mock("electron", () => ({ contextBridge: { exposeInMainWorld: host.expose },
  ipcRenderer: { invoke: host.invoke, on: vi.fn(), removeListener: vi.fn() }, webUtils: { getPathForFile: vi.fn() } }));

it("exposes exact decoder intents without worker execution, paths or caller-supplied review bindings", async () => {
  await import("../src/preload/index");
  const api = host.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi;
  expect(Object.keys(api.decoderExecution)).toEqual(["captureNext", "prepare", "calibrate", "apply", "status", "stop"]);
  await api.decoderExecution.captureNext({ brainId: "fixture" });
  expect(host.invoke).toHaveBeenLastCalledWith(IPC.decoderExecution.captureNext, { brainId: "fixture" });
  await api.decoderExecution.prepare({ brainId: "fixture" });
  expect(host.invoke).toHaveBeenLastCalledWith(IPC.decoderExecution.prepare, { brainId: "fixture" });
  await api.decoderExecution.calibrate({ brainId: "fixture", reviewId: "opaque-main-review" });
  expect(host.invoke).toHaveBeenLastCalledWith(IPC.decoderExecution.calibrate, { brainId: "fixture", reviewId: "opaque-main-review" });
  await api.decoderExecution.apply({ brainId: "fixture", proposalId: "owned-proposal", proposalSha256: "a".repeat(64) });
  expect(host.invoke).toHaveBeenLastCalledWith(IPC.decoderExecution.apply, { brainId: "fixture", proposalId: "owned-proposal", proposalSha256: "a".repeat(64) });
  await api.decoderExecution.status("fixture"); expect(host.invoke).toHaveBeenLastCalledWith(IPC.decoderExecution.status, "fixture");
  await api.decoderExecution.stop({ brainId: "fixture", operationId: "owned-operation" });
  expect(host.invoke).toHaveBeenLastCalledWith(IPC.decoderExecution.stop, { brainId: "fixture", operationId: "owned-operation" });
  expect(api.decoderExecution).not.toHaveProperty("execute"); expect(api.decoderExecution).not.toHaveProperty("load");
});
