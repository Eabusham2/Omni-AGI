import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  interactivePanelErrorResponse,
  interactivePanelHostReady,
  interactivePanelIdentity,
  interactivePanelRequestFromMessage,
  interactivePanelResponse,
  isCurrentInteractivePanelOperation,
  nextPendingInteractivePanelReviews
} from "../src/renderer/src/pluginInteractivePanelBridge";
import type { PluginInteractivePanelLaunch, PluginInteractiveToolReceipt } from "../src/shared/pluginInteractivePanels";

const launch: PluginInteractivePanelLaunch = {
  format: "omni-plugin-panel-launch", version: 1, viewId: "a".repeat(32), channelNonce: "b".repeat(64),
  brainId: "brain", pluginId: "example.canvas", contentSha256: "c".repeat(64), catalogRevision: 3,
  instanceRevision: 4, panelId: "canvas", title: "Canvas", url: `omni-plugin-app://${"a".repeat(32)}/index.html`
};

describe("interactive plugin Studio relay", () => {
  it("requires the exact opaque child source, nonce and schema", () => {
    const source = {} as MessageEventSource;
    const data = { format: "omni-plugin-panel-message", version: 1, viewId: launch.viewId,
      channelNonce: launch.channelNonce, requestId: "request-one", toolId: "reader", action: "read",
      arguments: { path: "note.txt" } };
    expect(interactivePanelRequestFromMessage({ source, origin: "null", data }, source, launch)?.request.arguments)
      .toEqual({ path: "note.txt" });
    expect(interactivePanelRequestFromMessage({ source: {} as MessageEventSource, origin: "null", data }, source, launch)).toBeNull();
    expect(interactivePanelRequestFromMessage({ source, origin: "file://", data }, source, launch)).toBeNull();
    expect(interactivePanelRequestFromMessage({ source, origin: "null", data: { ...data, channelNonce: "d".repeat(64) } }, source, launch)).toBeNull();
    expect(interactivePanelRequestFromMessage({ source, origin: "null", data: { ...data, approvalToken: "forged" } }, source, launch)).toBeNull();
  });

  it("keeps approval identity and review operands out of child responses", () => {
    const receipt: PluginInteractiveToolReceipt = {
      format: "omni-plugin-panel-receipt", version: 1, viewId: launch.viewId, requestId: "request-one",
      executionId: "execution", toolId: "plugin.example.canvas.reader", action: "read",
      state: "approval-required", startedAt: new Date(0).toISOString(), reviewId: "private-review",
      review: { kind: "plugin-nested-operation", pluginId: "example.canvas", pluginName: "Canvas",
        contentSha256: launch.contentSha256, outer: { toolId: "plugin.example.canvas.reader", action: "read" },
        resolved: { toolId: "system.files", action: "read", arguments: { path: "/private" } },
        permissionSources: [{ toolId: "plugin.example.canvas.reader", level: "full", inherited: false },
          { toolId: "system.files", level: "ask", inherited: true }] }
    };
    const response = interactivePanelResponse(launch, receipt);
    expect(JSON.stringify(response)).not.toContain("private-review");
    expect(JSON.stringify(response)).not.toContain("/private");
    expect(interactivePanelHostReady(launch)).toMatchObject({ kind: "ready", channelNonce: launch.channelNonce });
    expect(interactivePanelErrorResponse(launch, "request-one", new Error("bounded"))).toMatchObject({ error: "bounded" });
  });

  it("mounts only an opaque no-preload child and no direct executor", () => {
    const source = readFileSync(resolve(process.cwd(), "src/renderer/src/PluginInteractivePanel.tsx"), "utf8");
    const manager = readFileSync(resolve(process.cwd(), "src/renderer/src/PluginManager.tsx"), "utf8");
    expect(source).toContain('sandbox="allow-scripts"');
    expect(source).not.toContain("allow-same-origin");
    expect(source).not.toContain("srcDoc=");
    expect(source).not.toContain("dangerouslySetInnerHTML");
    expect(source).not.toMatch(/\beval\s*\(/);
    expect(source).toContain("api.panelRequest");
    expect(source).toContain("api.reviewPanelRequest");
    expect(source).toContain("Host action ·");
    expect(manager).toContain("<PluginInteractivePanel");
  });

  it("fences stale async owners and retains every concurrent Ask review", () => {
    const source = readFileSync(resolve(process.cwd(), "src/renderer/src/PluginInteractivePanel.tsx"), "utf8");
    const identity = interactivePanelIdentity(launch);
    expect(isCurrentInteractivePanelOperation({ mounted: true, currentIdentity: identity,
      expectedIdentity: identity, currentViewId: launch.viewId, expectedViewId: launch.viewId })).toBe(true);
    expect(isCurrentInteractivePanelOperation({ mounted: false, currentIdentity: identity,
      expectedIdentity: identity })).toBe(false);
    expect(isCurrentInteractivePanelOperation({ mounted: true, currentIdentity: identity,
      expectedIdentity: interactivePanelIdentity({ ...launch, brainId: "other" }) })).toBe(false);
    expect(isCurrentInteractivePanelOperation({ mounted: true, currentIdentity: identity,
      expectedIdentity: identity, currentViewId: launch.viewId, expectedViewId: "d".repeat(32) })).toBe(false);

    const review = (requestId: string): PluginInteractiveToolReceipt => ({
      format: "omni-plugin-panel-receipt", version: 1, viewId: launch.viewId, requestId,
      executionId: `execution-${requestId}`, toolId: "plugin.example.canvas.reader", action: "read",
      state: "approval-required", startedAt: new Date(0).toISOString(), reviewId: `review-${requestId}`,
      review: { kind: "plugin-nested-operation", pluginId: "example.canvas", pluginName: "Canvas",
        contentSha256: launch.contentSha256, outer: { toolId: "plugin.example.canvas.reader", action: "read" },
        resolved: { toolId: "system.files", action: "read", arguments: { path: requestId } },
        permissionSources: [{ toolId: "plugin.example.canvas.reader", level: "full", inherited: false },
          { toolId: "system.files", level: "ask", inherited: true }] }
    });
    const first = review("one"), second = review("two");
    const both = nextPendingInteractivePanelReviews(nextPendingInteractivePanelReviews([], first), second);
    expect(both.map((item) => item.requestId)).toEqual(["one", "two"]);
    const completedFirst = { ...first, state: "complete" as const, review: undefined, reviewId: undefined,
      finishedAt: new Date(1).toISOString() };
    expect(nextPendingInteractivePanelReviews(both, completedFirst).map((item) => item.requestId)).toEqual(["two"]);
    expect(source).toContain("await api.closePanel(opened.brainId");
    expect(source).toContain("pendingReviews.map");
  });
});
