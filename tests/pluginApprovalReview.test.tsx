import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { PluginApprovalReviewView, PluginToolApprovalReviewView, pluginApprovalReview, toolExecutionPluginApprovalReview } from "../src/renderer/src/PluginApprovalReview";
import type { ActionEvent } from "../src/shared/types";

function event(): ActionEvent {
  const now = new Date(0).toISOString();
  return {
    id: "event", brainId: "brain", createdAt: now, updatedAt: now, state: "approval-required",
    action: { kind: "tool", source: "brain", toolId: "plugin.example.notes.reader", action: "read", arguments: { path: "/outer" } },
    execution: { id: "execution", toolId: "plugin.example.notes.reader", action: "read", state: "approval-required", startedAt: now,
      approvalToken: "token", output: { kind: "plugin-nested-operation", pluginId: "example.notes", pluginName: "Notes helper",
        contentSha256: "a".repeat(64), outer: { toolId: "plugin.example.notes.reader", action: "read" },
        resolved: { toolId: "system.files", action: "read", arguments: { path: "/resolved" } },
        permissionSources: [{ toolId: "plugin.example.notes.reader", level: "full", inherited: false },
          { toolId: "system.files", level: "ask", inherited: true }] } }
  };
}

describe("plugin Ask review", () => {
  it("shows the resolved nested operation rather than only the opaque plugin name", () => {
    expect(pluginApprovalReview(event())?.resolved).toEqual({ toolId: "system.files", action: "read", arguments: { path: "/resolved" } });
    const html = renderToStaticMarkup(<PluginApprovalReviewView event={event()} />);
    expect(html).toContain("system.files.read");
    expect(html).toContain("/resolved");
    expect(html).toContain("both plugin and nested permissions");
    const manual = renderToStaticMarkup(<PluginToolApprovalReviewView execution={event().execution}
      toolId="plugin.example.notes.reader" action="read" />);
    expect(manual).toContain("system.files.read");
  });

  it("rejects malicious non-plugin and mismatched approval projections", () => {
    const valid = event();
    const forgedNonPlugin = structuredClone(valid);
    forgedNonPlugin.action.toolId = "mcp.hostile.tool";
    forgedNonPlugin.execution!.toolId = "mcp.hostile.tool";
    expect(pluginApprovalReview(forgedNonPlugin)).toBeNull();

    const mismatchedOuter = structuredClone(valid);
    (mismatchedOuter.execution!.output as { outer: { toolId: string } }).outer.toolId = "plugin.other.tool";
    expect(pluginApprovalReview(mismatchedOuter)).toBeNull();

    const completed = structuredClone(valid.execution!);
    completed.state = "complete";
    expect(toolExecutionPluginApprovalReview(completed, {
      toolId: valid.action.toolId,
      action: valid.action.action
    })).toBeNull();
  });
});
