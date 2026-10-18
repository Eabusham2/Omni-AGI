import { describe, expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";

const bridge = vi.hoisted(() => ({ expose: vi.fn(), on: vi.fn(), off: vi.fn(), invoke: vi.fn() }));
vi.mock("electron", () => ({ contextBridge: { exposeInMainWorld: bridge.expose },
  ipcRenderer: { on: bridge.on, removeListener: bridge.off, invoke: bridge.invoke }, webUtils: { getPathForFile: vi.fn() } }));
import "../src/preload/index";

describe("typed HTML artifact preload boundary", () => {
  it("exposes only handle-bound open/review operations and no raw markup/file API", async () => {
    const api = bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi;
    const identity = { brainId: "brain", originalBrainId: "origin", artifactId: "a".repeat(32),
      sourceActionId: "action", contentSha256: "b".repeat(64) };
    await api.artifacts.open(identity);
    await api.artifacts.activate({ ...identity, expectedRevision: 2, acknowledged: true });
    const request = { format: "omni-html-artifact-message" as const, version: 1 as const,
      viewId: "c".repeat(32), channelNonce: "d".repeat(64), requestId: "request", toolId: "system.files",
      action: "read", arguments: { path: "note.txt" } };
    await api.artifacts.request(request);
    await api.artifacts.review({ viewId: request.viewId, channelNonce: request.channelNonce,
      requestId: request.requestId, reviewId: "review", decision: "deny" });
    await api.artifacts.close("brain", request.viewId, request.channelNonce);
    expect(bridge.invoke.mock.calls).toEqual(expect.arrayContaining([
      [IPC.artifacts.open, identity],
      [IPC.artifacts.activate, { ...identity, expectedRevision: 2, acknowledged: true }],
      [IPC.artifacts.request, request],
      [IPC.artifacts.review, { viewId: request.viewId, channelNonce: request.channelNonce,
        requestId: request.requestId, reviewId: "review", decision: "deny" }],
      [IPC.artifacts.close, "brain", request.viewId, request.channelNonce]
    ]));
    expect("html" in api.artifacts).toBe(false);
    expect("script" in api.artifacts).toBe(false);
    expect("approvalToken" in request).toBe(false);
  });
});
