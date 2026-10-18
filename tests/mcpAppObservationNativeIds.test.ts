/** Current native/host ID grammar. No Worker, neural constructor or Load. */
import { describe, expect, it } from "vitest";
import fixture from "./fixtures/native-app-opaque-head.json";
import type { McpAppObservationArtifact } from "../src/shared/mcpAppArtifacts";
import type { NativeAppObservationFrontier, NativeAppObservationHead } from "../src/shared/nativeAppObservation";
import { nativeAppObservationFrontierSha256 } from "../src/main/nativeAppObservation";
import { checkedAppObservationFrontier, checkedAppObservationResult } from "../src/main/mcpAppObservationReceipt";
import { validateMcpAppArtifact } from "../src/main/mcpAppArtifacts";

const retained = fixture.frontier as NativeAppObservationFrontier;
const uuid = "11111111-1111-4111-8111-111111111111";
function envelope(streamId: unknown): NativeAppObservationFrontier {
  const head = { ...retained.head, streamId } as NativeAppObservationHead;
  return { ...retained, head, frontierSha256: nativeAppObservationFrontierSha256(head) };
}
const artifact: McpAppObservationArtifact = { format: "omni-mcp-app-observation", version: 1, id: uuid, artifactId: uuid,
  captureSha256: "c".repeat(64), source: { brainId: retained.brainId, serverId: "owned-source", serverLabel: "Owned source", endpoint: "fixture:owned" },
  observedAt: "2026-10-13T12:00:00.000Z", reviewId: uuid, method: "ui/update-model-context",
  content: { sha256: retained.head!.dataSha256, bytes: 1, mimeType: "application/json" }, streamId: uuid, ordinal: 1, previousSha256: "0".repeat(64), untrusted: true };

describe("current native App frontier bounded opaque stream IDs", () => {
  it("accepts the actual retained native head/hash without rewriting its opaque app-stream identity", () => {
    expect(nativeAppObservationFrontierSha256(retained.head)).toBe(retained.frontierSha256);
    expect(checkedAppObservationFrontier(retained, retained.brainId)).toBe(retained);
    expect(retained.head!.streamId).toBe("app-stream");
  });
  it("accepts the actual inherited root ID only as a current native grammar fixture, not an App producer claim", () => {
    const id = fixture.inheritedGrammarExample.value;
    expect(id).toBe("ancestral-50bcb3baad27d8fdee107612a3fb17cf896769d9ac20dd2226b69e8619b6de5b");
    expect(fixture.inheritedGrammarExample.scope).toContain("do not inherit");
    const value = envelope(id); expect(checkedAppObservationFrontier(value, retained.brainId).head!.streamId).toBe(id);
  });
  it("uses true full-match ASCII native bounds, not JS dollar or String coercion", () => {
    for (const id of [undefined, null, 123, ["app-stream"], {}, "", "-first", ".first", "_first", "bad/path", "bad space",
      "valid\n", "valid\r", "valid\r\n", "valid\u2028", "valid\0", "é-first", "a".repeat(129)])
      expect(() => checkedAppObservationFrontier(envelope(id), retained.brainId), `rejected ${String(id)}`).toThrow(/invalid committed head/);
    for (const id of ["a", "0", "a".repeat(128), "Native_09.part-two", uuid]) expect(checkedAppObservationFrontier(envelope(id), retained.brainId).head!.streamId).toBe(id);
  });
  it("retains exact owner, native hash and reviewed record/CAS binding after the grammar fix", () => {
    expect(() => checkedAppObservationFrontier(retained, "another-brain")).toThrow(/another instance/);
    expect(() => checkedAppObservationFrontier({ ...retained, head: { ...retained.head!, streamId: "other-valid-stream" } }, retained.brainId)).toThrow(/checksum/);
    const value = envelope(uuid), result = { ...value, duplicate: false };
    expect(checkedAppObservationResult(result, retained.brainId, artifact, value.head!.recordSha256, "observe", value.head!.baseFrontierSha256)).toBe(result);
    expect(() => checkedAppObservationResult(result, retained.brainId, { ...artifact, streamId: "another-reviewed-stream" }, value.head!.recordSha256, "observe", value.head!.baseFrontierSha256)).toThrow(/exact reviewed/);
    expect(() => checkedAppObservationResult(result, retained.brainId, artifact, "f".repeat(64), "observe", value.head!.baseFrontierSha256)).toThrow(/exact reviewed/);
    expect(() => checkedAppObservationResult(result, retained.brainId, artifact, value.head!.recordSha256, "observe", "f".repeat(64))).toThrow(/exact reviewed/);
  });
  it("does not change any queued artifact, review, observation or stream UUID requirement", () => {
    expect(validateMcpAppArtifact(artifact)).toBe(artifact);
    for (const field of ["artifactId", "id", "reviewId", "streamId"])
      expect(() => validateMcpAppArtifact({ ...artifact, [field]: fixture.inheritedGrammarExample.value }), field).toThrow(/Invalid inert|invalid task/);
  });
});
