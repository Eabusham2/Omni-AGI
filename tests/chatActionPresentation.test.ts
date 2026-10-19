import { describe, expect, it } from "vitest";
import {
  chatActionSearchActivity,
  chatActionMcpAppViewId,
  chatActionStateLabel,
  chatActionTitle,
  chatApprovalStatus,
  cleanChatActionStatus
} from "../src/renderer/src/chatActionPresentation";
import type { ActionEvent, StructuredAction } from "../src/shared/types";

const ponder: StructuredAction = {
  kind: "ponder",
  source: "organic",
  toolId: "ponder",
  action: "ponder",
  arguments: {}
};

describe("plain chat action presentation", () => {
  it("labels a pending approval with its own action and live readable expiry", () => {
    expect(chatApprovalStatus("Web access proposed in chat", new Date(3598_000).toISOString(), 0))
      .toBe("Web access · needs approval · 59m 58s remaining");
    expect(chatApprovalStatus("Web access", new Date(3598_000).toISOString(), 1_000))
      .toBe("Web access · needs approval · 59m 57s remaining");
    expect(chatApprovalStatus("File access", new Date(3601_000).toISOString(), 0))
      .toBe("File access · needs approval · 1h 0m 1s remaining");
    expect(chatApprovalStatus("File access", undefined, 0)).toBe("File access · needs approval");
    expect(chatApprovalStatus("File access", "invalid", 0)).toBe("File access · needs approval");
  });
  const connectedApp = (): ActionEvent => ({ id: "action", brainId: "brain", state: "complete", createdAt: "2026-10-11T00:00:00Z", updatedAt: "2026-10-11T00:00:00Z",
    action: { kind: "tool", source: "brain", toolId: "mcp.server.fixture.hash", action: "call", arguments: {} },
    execution: { id: "execution", toolId: "mcp.server.fixture.hash", action: "call", state: "complete", startedAt: "2026-10-11T00:00:00Z",
      output: { _meta: { "omni/mcpApp": { id: "opaque-view-id", brainId: "brain", toolId: "mcp.server.fixture.hash", untrusted: true } } } } });
  it("shows only a matching completed MCP app hint and leaves authority to main-owned lookup", () => {
    const event = connectedApp(); expect(chatActionMcpAppViewId(event)).toBe("opaque-view-id");
    expect(chatActionMcpAppViewId({ ...event, state: "running" })).toBeUndefined();
    expect(chatActionMcpAppViewId({ ...event, execution: { ...event.execution!, state: "failed" } })).toBeUndefined();
    expect(chatActionMcpAppViewId({ ...event, execution: { ...event.execution!, toolId: "different-tool" } })).toBeUndefined();
    expect(chatActionMcpAppViewId({ ...event, brainId: "other" })).toBeUndefined();
  });
  it("does not treat malformed metadata, arbitrary URLs or a model's proposed action as a view", () => {
    const event = connectedApp();
    for (const descriptor of [null, [], { id: "https://untrusted.example/", brainId: "brain", toolId: event.action.toolId, untrusted: true },
      { id: "opaque", brainId: "brain", toolId: "different-tool", untrusted: true }, { id: "opaque", brainId: "brain", toolId: event.action.toolId, untrusted: false }]) {
      expect(chatActionMcpAppViewId({ ...event, execution: { ...event.execution!, output: { _meta: { "omni/mcpApp": descriptor } } } })).toBeUndefined();
    }
    expect(chatActionMcpAppViewId({ ...event, state: "proposed", execution: undefined })).toBeUndefined();
  });
  it("prefers a verified saved-artifact owner projection without rewriting the original result", () => {
    const event = connectedApp();
    const original = JSON.stringify(event.execution);
    const copied: ActionEvent = { ...event, brainId: "copy", mcpAppView: { id: "copied-view", brainId: "copy", serverId: "server", serverLabel: "Fixture",
      resourceUri: "ui://fixture/editor", toolId: event.action.toolId!, remoteToolName: "editor", observedAt: event.createdAt,
      untrusted: true, savedArtifact: true, artifactSha256: "a".repeat(64), needsMappingReview: true,
      originalSource: { brainId: "brain", serverId: "server", serverLabel: "Original fixture", endpoint: "https://fixture.invalid/mcp" } } };
    expect(chatActionMcpAppViewId(copied)).toBe("copied-view");
    expect(JSON.stringify(copied.execution)).toBe(original);
    expect(chatActionMcpAppViewId({ ...copied, mcpAppView: { ...copied.mcpAppView!, brainId: "other" } })).toBeUndefined();
    expect(chatActionMcpAppViewId({ ...copied, mcpAppView: { ...copied.mcpAppView!, artifactSha256: "not-a-hash" } })).toBeUndefined();
  });
  it("never exposes legacy ponder or creativity protocol strings", () => {
    expect(chatActionTitle(ponder)).toBe("Pondering");
    expect(cleanChatActionStatus("ponder.ponder: running.", ponder)).toBe(
      "Pondering: running."
    );
    expect(cleanChatActionStatus("studio.ui/open-creativity: complete.", {
      ...ponder,
      kind: "tool",
      toolId: "studio.ui",
      action: "open-creativity"
    })).toBe("Creativity: complete.");
  });

  it("uses human action and state names", () => {
    expect(chatActionTitle({
      ...ponder,
      kind: "imagine",
      toolId: "modality.imagine",
      action: "generate",
      arguments: { modality: "video" }
    })).toBe("Video imagination");
    expect(chatActionStateLabel("approval-required")).toBe("Needs approval");
  });

  it("shows the exact bounded web search in the collapsed activity card", () => {
    const action = {
      kind: "tool" as const,
      toolId: "web.search",
      action: "search",
      arguments: { query: "liquid neural network stability" },
      source: "brain" as const
    };
    expect(chatActionSearchActivity(action, "running")).toBe(
      "Searching “liquid neural network stability”"
    );
    expect(chatActionSearchActivity(action, "complete")).toBe(
      "Searched “liquid neural network stability”"
    );
  });

});
