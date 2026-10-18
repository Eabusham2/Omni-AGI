import { expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";
const bridge = vi.hoisted(() => ({ expose: vi.fn(), invoke: vi.fn(), on: vi.fn(), off: vi.fn() }));
vi.mock("electron", () => ({ contextBridge: { exposeInMainWorld: bridge.expose }, ipcRenderer: { invoke: bridge.invoke, on: bridge.on, removeListener: bridge.off }, webUtils: { getPathForFile: vi.fn() } }));
import "../src/preload/index";
it("exposes only human capture controls/read views, not native frames, output leases or provenance authority", async () => {
  const capture = (bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi).modality.capture;
  await capture.status("brain"); await capture.record({ brainId: "brain", requestId: "record", lookbackSeconds: 30 });
  await capture.resume({ brainId: "brain", recordingId: "saved", expectedRevision: 3 }); await capture.fragments("brain", "saved", { afterIndex: 5, limit: 10 });
  expect(Object.keys(capture).sort()).toEqual(["cancel", "fragments", "onEvent", "record", "recordings", "resume", "settings", "status", "stop"]);
  expect(bridge.invoke.mock.calls).toEqual(expect.arrayContaining([[IPC.modality.captureStatus, "brain"], [IPC.modality.captureRecord, { brainId: "brain", requestId: "record", lookbackSeconds: 30 }],
    [IPC.modality.captureResume, { brainId: "brain", recordingId: "saved", expectedRevision: 3 }], [IPC.modality.captureFragments, "brain", "saved", { afterIndex: 5, limit: 10 }]]));
  const listener = vi.fn(), stop = capture.onEvent(listener), wrapped = bridge.on.mock.calls.find(([name]) => name === IPC.modality.captureEvent)?.[1]; wrapped({}, { brainId: "brain", type: "changed" }); expect(listener).toHaveBeenCalledTimes(1);
  stop(); expect(bridge.off).toHaveBeenCalledWith(IPC.modality.captureEvent, wrapped);
});
