import { describe, expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";

const bridge = vi.hoisted(() => ({ expose: vi.fn(), on: vi.fn(), off: vi.fn(), invoke: vi.fn() }));
vi.mock("electron", () => ({
  contextBridge: { exposeInMainWorld: bridge.expose },
  ipcRenderer: { on: bridge.on, removeListener: bridge.off, invoke: bridge.invoke },
  webUtils: { getPathForFile: vi.fn() }
}));
import "../src/preload/index";

describe("typed plugin preload surface", () => {
  it("exposes catalog/maker/access operations without filesystem or executor objects", async () => {
    const api = bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi;
    await api.plugins.list();
    await api.plugins.package("example.notes", "a".repeat(64));
    await api.plugins.setEnabled("example.notes", true, "a".repeat(64), 7);
    await api.plugins.instance("brain-one", "example.notes");
    await api.plugins.portable("brain-one");
    const panelRequest = { format: "omni-plugin-panel-message" as const, version: 1 as const,
      viewId: "b".repeat(32), channelNonce: "c".repeat(64), requestId: "request-one",
      toolId: "reader", action: "read", arguments: { path: "note.txt" } };
    await api.plugins.openPanel("brain-one", "example.notes", "a".repeat(64), "canvas");
    await api.plugins.panelRequest(panelRequest);
    await api.plugins.reviewPanelRequest({ viewId: "b".repeat(32), channelNonce: "c".repeat(64),
      requestId: "request-one", reviewId: "review-one", decision: "deny" });
    await api.plugins.closePanel("brain-one", "b".repeat(32), "c".repeat(64));
    const listener = vi.fn();
    const unsubscribe = api.plugins.onEvent(listener);
    const event = { type: "semantic-learning" as const, pluginId: "example.notes",
      contentSha256: "a".repeat(64), catalogRevision: 3, brainId: "brain-one",
      semanticState: "complete" as const, at: "2026-10-11T00:00:00.000Z" };
    const wrapped = bridge.on.mock.calls.find(([channel]) => channel === IPC.plugins.event)?.[1];
    wrapped?.({}, event);
    expect(listener).toHaveBeenCalledWith(event);
    unsubscribe();
    expect(bridge.off).toHaveBeenCalledWith(IPC.plugins.event, wrapped);
    expect(bridge.invoke.mock.calls).toEqual(expect.arrayContaining([
      [IPC.plugins.list],
      [IPC.plugins.package, "example.notes", "a".repeat(64)],
      [IPC.plugins.setEnabled, "example.notes", true, "a".repeat(64), 7],
      [IPC.plugins.instance, "brain-one", "example.notes"],
      [IPC.plugins.portable, "brain-one"],
      [IPC.plugins.openPanel, "brain-one", "example.notes", "a".repeat(64), "canvas"],
      [IPC.plugins.panelRequest, panelRequest],
      [IPC.plugins.reviewPanelRequest, { viewId: "b".repeat(32), channelNonce: "c".repeat(64),
        requestId: "request-one", reviewId: "review-one", decision: "deny" }],
      [IPC.plugins.closePanel, "brain-one", "b".repeat(32), "c".repeat(64)]
    ]));
    expect("execute" in api.plugins).toBe(false);
    expect("readFile" in api.plugins).toBe(false);
    expect("credentials" in api.plugins).toBe(false);
    expect("approvalToken" in panelRequest).toBe(false);
  });
});
