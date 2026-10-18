import { expect, it } from "vitest";
import { HtmlArtifactView } from "../src/renderer/src/HtmlArtifactView";
import { htmlArtifactIdentity, isCurrentHtmlArtifactOperation } from "../src/renderer/src/htmlArtifactPresentation";
import type { HtmlArtifactsApi } from "../src/shared/htmlArtifacts";
import type { ActionEvent } from "../src/shared/types";

function event(brainId = "brain-one", contentSha256 = "b".repeat(64)): ActionEvent {
  const now = new Date(0).toISOString();
  return { id: "action-one", brainId, state: "complete", createdAt: now, updatedAt: now,
    action: { kind: "tool", source: "brain", toolId: "studio.artifact", action: "create", arguments: {} },
    execution: { id: "execution", toolId: "studio.artifact", action: "create", state: "complete", startedAt: now,
      finishedAt: now, output: { format: "omni-chat-html-artifact-result", version: 1, artifact: {
        format: "omni-chat-html-artifact-projection", version: 1, brainId: "brain-one", originalBrainId: "brain-one",
        artifactId: "a".repeat(32), sourceActionId: "action-one", contentSha256, title: "Chart",
        interactive: true, operationalState: "enabled", revision: 1, createdAt: now, savedArtifact: true, untrusted: true
      } } }
  };
}

it("keys each mounted artifact instance only from its validated saved identity", () => {
  const api = {} as HtmlArtifactsApi, original = HtmlArtifactView({ event: event(), api });
  expect(original?.key).toBe(htmlArtifactIdentity({ brainId: "brain-one", originalBrainId: "brain-one",
    artifactId: "a".repeat(32), sourceActionId: "action-one", contentSha256: "b".repeat(64) }));
  const displayOnly = event();
  ((displayOnly.execution!.output as Record<string, unknown>).artifact as Record<string, unknown>).title = "Renamed display";
  expect(HtmlArtifactView({ event: displayOnly, api })?.key).toBe(original?.key);
  expect(HtmlArtifactView({ event: event("brain-two"), api })?.key).not.toBe(original?.key);
  expect(HtmlArtifactView({ event: event("brain-one", "c".repeat(64)), api })?.key).not.toBe(original?.key);
  const invalid = event(); invalid.action.toolId = "web.fetch";
  expect(HtmlArtifactView({ event: invalid, api })).toBeNull();
});

it("rejects a late activation result after unmount, identity/view replacement, or a newer open", () => {
  const identity = htmlArtifactIdentity({ brainId: "brain-one", originalBrainId: "brain-one",
    artifactId: "a".repeat(32), sourceActionId: "action-one", contentSha256: "b".repeat(64) });
  const current = { mounted: true, currentIdentity: identity, expectedIdentity: identity,
    currentOpenEpoch: 3, expectedOpenEpoch: 3, currentViewId: "view-one", expectedViewId: "view-one" };
  expect(isCurrentHtmlArtifactOperation(current)).toBe(true);
  expect(isCurrentHtmlArtifactOperation({ ...current, mounted: false })).toBe(false);
  expect(isCurrentHtmlArtifactOperation({ ...current, currentIdentity: `${identity}-next` })).toBe(false);
  expect(isCurrentHtmlArtifactOperation({ ...current, currentViewId: "view-two" })).toBe(false);
  expect(isCurrentHtmlArtifactOperation({ ...current, currentOpenEpoch: 4 })).toBe(false);
});
