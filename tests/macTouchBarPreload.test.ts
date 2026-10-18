import { describe, expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";
import { TOUCH_BAR_STATE_FORMAT, type TouchBarActionEvent } from "../src/shared/touchBar";

const bridge = vi.hoisted(() => ({ expose: vi.fn(), on: vi.fn(), off: vi.fn(), invoke: vi.fn() }));
vi.mock("electron", () => ({ contextBridge: { exposeInMainWorld: bridge.expose },
  ipcRenderer: { on: bridge.on, removeListener: bridge.off, invoke: bridge.invoke }, webUtils: { getPathForFile: vi.fn() } }));
import "../src/preload/index";

describe("typed Touch Bar preload seam", () => {
  it("sends only state and exposes a removable typed action listener", async () => {
    const api = bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi;
    const state = { format: TOUCH_BAR_STATE_FORMAT, version: 1 as const, revision: 3, brainId: "brain-one",
      instanceName: "Ada", phase: "conversation" as const, submit: { mode: "queue" as const, enabled: true } };
    await api.window.touchBar.update(state);
    expect(bridge.invoke).toHaveBeenCalledWith(IPC.window.touchBarUpdate, state);
    const listener = vi.fn(), dispose = api.window.touchBar.onAction(listener);
    const wrapped = bridge.on.mock.calls.find(([channel]) => channel === IPC.window.touchBarAction)?.[1];
    const event: TouchBarActionEvent = { format: "omni-touch-bar-action", version: 1, brainId: "brain-one",
      stateRevision: 3, action: "submit", submitMode: "queue" };
    wrapped({}, event); expect(listener).toHaveBeenCalledWith(event); dispose();
    expect(bridge.off).toHaveBeenCalledWith(IPC.window.touchBarAction, wrapped);
    expect(JSON.stringify(state)).not.toContain("prompt");
  });
});
