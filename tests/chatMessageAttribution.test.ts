import { describe, expect, it } from "vitest";
import { chatMessageAttribution } from "../src/renderer/src/chatMessageAttribution";
import type { ChatMessage } from "../src/shared/types";
import type { ChatInputProvenance } from "../src/shared/chatInputProvenance";

const source: ChatInputProvenance = {
  kind: "mcp-app", sourceBrainId: "source", serverId: "editor", serverLabel: "Editor",
  endpoint: "https://fixture.invalid/mcp", resourceUri: "ui://fixture/editor", artifactId: "artifact",
  captureSha256: "a".repeat(64), originalCallBrainId: "original", toolId: "mcp.editor.render",
  reviewId: "review", observationId: "observation", streamId: "stream", ordinal: 1,
  recordSha256: "b".repeat(64), dataSha256: "c".repeat(64), nativeFrontierSha256: "d".repeat(64), inputSha256: "e".repeat(64)
};
const message = (inputProvenance?: ChatInputProvenance): ChatMessage => ({
  id: "message", role: "human", content: "The exact reviewed task.", createdAt: "2026-10-11T00:00:00Z",
  ...(inputProvenance ? { inputProvenance } : {})
});
describe("connected-App chat attribution", () => {
  it("never displays an attributed App task as authored by the human", () => {
    expect(chatMessageAttribution(message(source), "Cortex")).toEqual({ name: "Editor", connectedApp: true, verifiedShape: true });
    expect(chatMessageAttribution(message(), "Cortex").name).toBe("You");
  });
  it("keeps the brain label and exact task/source bytes untouched", () => {
    const original = message({ ...source, serverLabel: "\u202eEditor" });
    expect(chatMessageAttribution(original, "Cortex").name).toBe("Editor");
    expect(original.inputProvenance!.serverLabel).toBe("\u202eEditor");
    expect(original.content).toBe("The exact reviewed task.");
    expect(chatMessageAttribution({ ...original, role: "brain" }, "Cortex").name).toBe("Cortex");
  });
  it("does not turn malformed source metadata into a human attribution", () => {
    expect(chatMessageAttribution(message({ ...source, recordSha256: "invalid" }), "Cortex"))
      .toEqual({ name: "Unverified app source", connectedApp: true, verifiedShape: false });
  });
});
