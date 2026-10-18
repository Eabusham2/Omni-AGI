import { describe, expect, it, vi } from "vitest";
import { IPC } from "../src/shared/ipc";
import type { OmniApi } from "../src/shared/types";
const bridge = vi.hoisted(() => ({ expose: vi.fn(), invoke: vi.fn(), on: vi.fn(), off: vi.fn() }));
vi.mock("electron", () => ({ contextBridge: { exposeInMainWorld: bridge.expose },
  ipcRenderer: { invoke: bridge.invoke, on: bridge.on, removeListener: bridge.off }, webUtils: { getPathForFile: vi.fn() } }));
import "../src/preload/index";

describe("read/dismiss-only steering preload boundary", () => {
  it("exposes typed paged/exact reads, display CAS and removable invalidation events; no edit/resend/execute API", async () => {
    const api = (bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi).chat.directions;
    await api.list("owner", { beforeSequence: 20, limit: 10 }); await api.get("owner", "direction-id"); await api.forTurns("owner", ["visible-turn"]);
    const request = { brainId: "owner", id: "direction-id", expectedRevision: 3, dismissed: true }; await api.dismiss(request);
    expect(bridge.invoke.mock.calls).toEqual(expect.arrayContaining([[IPC.chat.directionsList, "owner", { beforeSequence: 20, limit: 10 }],
      [IPC.chat.directionsGet, "owner", "direction-id"], [IPC.chat.directionsForTurns, "owner", ["visible-turn"]], [IPC.chat.directionsDismiss, request]]));
    expect(Object.keys(api).sort()).toEqual(["dismiss", "forTurns", "get", "list", "onEvent"]);
    const listener = vi.fn(), dispose = api.onEvent(listener), wrapped = bridge.on.mock.calls.find(([event]) => event === IPC.chat.directionsEvent)?.[1];
    wrapped({}, { brainId: "owner", id: "direction-id", revision: 4 }); expect(listener).toHaveBeenCalledWith({ brainId: "owner", id: "direction-id", revision: 4 });
    dispose(); expect(bridge.off).toHaveBeenCalledWith(IPC.chat.directionsEvent, wrapped);
  });
});
