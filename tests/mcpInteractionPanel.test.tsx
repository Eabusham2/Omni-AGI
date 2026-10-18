import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { McpInteractionCard, McpInteractionPanel } from "../src/renderer/src/McpInteractionPanel";
import { McpAppObservationCard, McpAppView } from "../src/renderer/src/McpAppView";
import type { McpInteractionRecord } from "../src/shared/mcpInteractions";
import type { McpAppObservationReceipt } from "../src/shared/mcpAppTaskData";
import { APP_AUDIT_PNG, APP_AUDIT_WAV, syntheticAppAudit } from "./fixtures/mcpAppAudit";

const base = { format: "omni-mcp-interaction" as const, version: 1 as const, id: "fixture", status: "pending" as const,
  createdAt: "2026-10-11T00:00:00Z", updatedAt: "2026-10-11T00:00:00Z",
  origin: { brainId: "first", serverId: "app", serverLabel: "Synthetic app", endpoint: "https://fixture.invalid/mcp", connectionEpoch: 1 } };
describe("MCP review rendering (static component fixtures, not visual acceptance)", () => {
  it("keeps media as reviewed metadata rather than decoding/playing untrusted bytes", () => {
    const record: McpInteractionRecord = { ...base, kind: "sampling-input", request: { brainId: "first", requestId: "sample", maxTokens: 32,
      messages: [{ role: "user", content: [{ type: "text", text: "Reviewed task" }, { type: "image", mimeType: "image/png", data: "UNTRUSTED_RAW_BASE64_FIXTURE" }] }] },
      media: [{ messageIndex: 0, blockIndex: 1, kind: "image", mimeType: "image/png", decodedBytes: 68, encodedSha256: "a".repeat(64) }] };
    const html = renderToStaticMarkup(<McpInteractionCard record={record} onRespond={async () => undefined} />);
    expect(html).toContain("image/png"); expect(html).toContain("68"); expect(html).toContain("Reviewed task");
    expect(html).not.toContain("UNTRUSTED_RAW_BASE64_FIXTURE"); expect(html).not.toContain("<img"); expect(html).not.toContain("<audio");
    expect(html).toContain("button button--primary"); expect(html).toContain("Allow private inference");
  });
  it("uses a scrollable exact-origin App review instead of an oversized native alert", () => {
    const record: McpInteractionRecord = { ...base, kind: "app-permission", request: { category: "network", message: "Review these origins", csp: {
      connectDomains: ["https://api.example"], resourceDomains: [], frameDomains: [], baseUriDomains: [] } } };
    const html = renderToStaticMarkup(<McpInteractionCard record={record} onRespond={async () => undefined} />);
    expect(html).toContain("Exact network origins"); expect(html).toContain("https://api.example");
    expect(html).toContain("Allow reviewed request"); expect(html).toContain("does not change the instance");
    expect(html).not.toContain("type=\"password\""); expect(html).not.toContain("assumed");
  });
  it("does not expose fake controls or fetch an App when the preview lacks a bridge", () => {
    expect(renderToStaticMarkup(<McpInteractionPanel brainId="first" />)).toBe("");
    expect(renderToStaticMarkup(<McpAppView brainId="first" viewId="view" />)).toBe("");
    const api = { describe: vi.fn(), open: vi.fn(), request: vi.fn(), close: vi.fn() };
    const html = renderToStaticMarkup(<McpAppView brainId="first" viewId="view" api={api} />);
    expect(html).toContain("Open interactive app"); expect(html).not.toContain("<iframe"); expect(api.open).not.toHaveBeenCalled();
  });
  it("labels App-proposed data as pending and untrusted, never fake user authorship or decoded remote media", () => {
    const record: McpInteractionRecord = { ...base, kind: "app-data", request: { message: "Review App-proposed task", captureSha256: "c".repeat(64), dataSha256: "d".repeat(64),
      view: { id: "view", brainId: "first", serverId: "app", serverLabel: "Synthetic app", resourceUri: "ui://fixture/view", toolId: "mcp.first.app.open", remoteToolName: "open", observedAt: base.createdAt, untrusted: true },
      data: { method: "ui/update-model-context", params: { content: [{ type: "text", text: "<script>untrusted task</script>" }, { type: "image", mimeType: "image/png", data: "UNTRUSTED_RAW_BASE64_FIXTURE" }], structuredContent: { userId: "opaque-user" } } },
      untrustedTaskData: true, hiddenPromptUsed: false } };
    const html = renderToStaticMarkup(<McpInteractionCard record={record} onRespond={async () => undefined} />);
    expect(html).toContain("Queue reviewed App data"); expect(html).toContain("pending task data"); expect(html).toContain("not from you");
    expect(html).toContain("&lt;script&gt;untrusted task&lt;/script&gt;"); expect(html).toContain("opaque-user"); expect(html).toContain("image/png");
    expect(html).not.toContain("UNTRUSTED_RAW_BASE64_FIXTURE"); expect(html).not.toContain("<img");
  });
  it("shows source-measured media coverage only after an actual-shaped native ACK, not while pending or as a copied live grant", () => {
    const receipt: McpAppObservationReceipt = { id: "fixture", artifactId: "view", recordSha256: "a".repeat(64), streamId: "stream", ordinal: 1,
      observedAt: base.createdAt, method: "ui/update-model-context", state: "pending-native-admission", neuralAdmission: "not-yet-admitted" };
    const pending = renderToStaticMarkup(<McpAppObservationCard observation={receipt} />); expect(pending).not.toContain("Input coverage");
    const content = syntheticAppAudit(JSON.stringify({ method: "ui/update-model-context", params: { content: [
      { type: "image", data: APP_AUDIT_PNG, mimeType: "image/png" }, { type: "audio", data: APP_AUDIT_WAV, mimeType: "audio/wav" }
    ] } }), receipt.recordSha256, "first").content;
    const admitted: McpAppObservationReceipt = { ...receipt, state: "admitted", neuralAdmission: "admitted", nativeReceipt: {
      format: "omni-native-app-observation-frontier", version: 1, brainId: "first", duplicate: false, frontierSha256: "d".repeat(64),
      head: { format: "omni-native-app-observation-head", version: 1, streamId: "stream", ordinal: 1, recordSha256: receipt.recordSha256,
        dataSha256: content.audit.dataSha256, previousSha256: "0".repeat(64), baseFrontierSha256: "0".repeat(64), disposition: "observe", state: "observed", committedAt: base.createdAt,
        learning: { kind: "experience", source: "mcp-app", sourceLabel: "Synthetic fixture", steps: 0, experienceCountBefore: 1, experienceCountAfter: 2,
          assemblyId: "synthetic-only", afterimageId: "", textSha256: "e".repeat(64), textBytes: 0 }, content } } };
    const html = renderToStaticMarkup(<McpAppObservationCard observation={admitted} />);
    expect(html).toContain("Input coverage: 2 blocks · 1 image frame · 4 audio samples"); expect(html).toContain("Recorded admission details");
    expect(html).not.toContain(APP_AUDIT_PNG); expect(html).not.toContain("<img"); expect(html).not.toContain("<audio");
    const copied = renderToStaticMarkup(<McpAppObservationCard observation={{ ...admitted, state: "dormant" }} />);
    expect(copied).toContain("Retained App data — no automatic admission");
  });
});
