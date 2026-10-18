import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  chatHtmlArtifactRequest,
  htmlArtifactError,
  htmlArtifactHostReady,
  htmlArtifactRequestFromMessage,
  htmlArtifactResponse,
  nextHtmlArtifactReviews
} from "../src/renderer/src/htmlArtifactPresentation";
import type { HtmlArtifactLaunch, HtmlArtifactToolReceipt } from "../src/shared/htmlArtifacts";
import type { ActionEvent } from "../src/shared/types";

function event(brainId = "brain-one"): ActionEvent {
  const now = new Date(0).toISOString();
  return { id: "action-one", brainId, state: "complete", createdAt: now, updatedAt: now,
    action: { kind: "tool", source: "brain", toolId: "studio.artifact", action: "create", arguments: {} },
    execution: { id: "execution", toolId: "studio.artifact", action: "create", state: "complete", startedAt: now,
      finishedAt: now, output: { format: "omni-chat-html-artifact-result", version: 1, artifact: {
        format: "omni-chat-html-artifact-projection", version: 1, brainId: "brain-one", originalBrainId: "brain-one",
        artifactId: "a".repeat(32), sourceActionId: "action-one", contentSha256: "b".repeat(64), title: "Chart",
        interactive: true, operationalState: "enabled", revision: 1, createdAt: now, savedArtifact: true, untrusted: true
      } } }
  };
}

const launch: HtmlArtifactLaunch = { format: "omni-chat-html-artifact-launch", version: 1,
  viewId: "c".repeat(32), channelNonce: "d".repeat(64), artifact: (event().execution!.output as any).artifact,
  url: `omni-html-artifact://${"c".repeat(32)}/index.html`, height: 360, scriptEnabled: true };

describe("chat HTML artifact presentation boundary", () => {
  it("projects only a completed real artifact tool result and keeps inherited origin identity", () => {
    expect(chatHtmlArtifactRequest(event())).toEqual({ brainId: "brain-one", originalBrainId: "brain-one",
      artifactId: "a".repeat(32), sourceActionId: "action-one", contentSha256: "b".repeat(64) });
    expect(chatHtmlArtifactRequest(event("copied-brain"))).toMatchObject({ brainId: "copied-brain", originalBrainId: "brain-one" });
    const forged = event(); forged.action.toolId = "web.fetch";
    expect(chatHtmlArtifactRequest(forged)).toBeNull();
    const changed = event(); (changed.execution!.output as any).artifact.contentSha256 = "invalid";
    expect(chatHtmlArtifactRequest(changed)).toBeNull();
  });

  it("requires the exact opaque frame source, nonce, and request schema", () => {
    const source = {} as MessageEventSource;
    const data = { format: "omni-html-artifact-message", version: 1, viewId: launch.viewId,
      channelNonce: launch.channelNonce, requestId: "request-one", toolId: "system.files", action: "read",
      arguments: { path: "note.txt" } };
    expect(htmlArtifactRequestFromMessage({ source, origin: "null", data }, source, launch)?.request.arguments).toEqual({ path: "note.txt" });
    expect(htmlArtifactRequestFromMessage({ source: {} as MessageEventSource, origin: "null", data }, source, launch)).toBeNull();
    expect(htmlArtifactRequestFromMessage({ source, origin: "file://", data }, source, launch)).toBeNull();
    expect(htmlArtifactRequestFromMessage({ source, origin: "null", data: { ...data, approvalToken: "forged" } }, source, launch)).toBeNull();
  });

  it("withholds review identity/operands from the child and retains concurrent Ask cards", () => {
    const review = (id: string): HtmlArtifactToolReceipt => ({ format: "omni-chat-html-artifact-receipt", version: 1,
      viewId: launch.viewId, requestId: id, executionId: `execution-${id}`, toolId: "system.files", action: "read",
      state: "approval-required", startedAt: new Date(0).toISOString(), reviewId: `review-${id}`,
      review: { kind: "html-artifact-operation", artifactId: launch.artifact.artifactId,
        contentSha256: launch.artifact.contentSha256,
        resolved: { toolId: "system.files", action: "read", arguments: { path: `/${id}` } } } });
    const first = review("one"), second = review("two"), both = nextHtmlArtifactReviews(nextHtmlArtifactReviews([], first), second);
    expect(both.map((item) => item.requestId)).toEqual(["one", "two"]);
    expect(JSON.stringify(htmlArtifactResponse(launch, first))).not.toContain("review-one");
    expect(JSON.stringify(htmlArtifactResponse(launch, first))).not.toContain("/one");
    expect(htmlArtifactHostReady(launch)).toMatchObject({ kind: "ready", channelNonce: launch.channelNonce });
    expect(htmlArtifactError(launch, "one", new Error("bounded"))).toMatchObject({ error: "bounded" });
  });

  it("keeps raw markup out of Studio and uses only an opaque iframe API", () => {
    const source = readFileSync(resolve(process.cwd(), "src/renderer/src/HtmlArtifactView.tsx"), "utf8");
    expect(source).toContain('sandbox={launch.scriptEnabled ? "allow-scripts" : ""}');
    expect(source).not.toContain("allow-same-origin");
    expect(source).not.toContain("srcDoc=");
    expect(source).not.toContain("dangerouslySetInnerHTML");
    expect(source).toContain("api.open(request)");
    expect(source).toContain("api.review(");
  });
});
