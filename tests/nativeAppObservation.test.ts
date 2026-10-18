import { describe, expect, it } from "vitest";
import type { NativeAppObservationHead } from "../src/shared/nativeAppObservation";
import { EMPTY_NATIVE_APP_OBSERVATION_FRONTIER } from "../src/shared/nativeAppObservation";
import { nativeAppObservationFrontierSha256, nativeAppObservationRawSha256 } from "../src/main/nativeAppObservation";

const head: NativeAppObservationHead = {
  format: "omni-native-app-observation-head", version: 1,
  streamId: "11111111-1111-1111-1111-111111111111", ordinal: 1,
  recordSha256: "a".repeat(64), dataSha256: "b".repeat(64), previousSha256: "0".repeat(64),
  baseFrontierSha256: "0".repeat(64), disposition: "observe", state: "observed",
  committedAt: "2026-10-11T12:00:00.000Z",
  learning: { kind: "experience", source: "mcp-app", sourceLabel: "reviewed-mcp-app:fixture/é",
    steps: 0, experienceCountBefore: 7, experienceCountAfter: 8, assemblyId: "assembly-one",
    afterimageId: "scratch-one", textSha256: "c".repeat(64), textBytes: 4 },
};

describe("native App observation exact-byte and frontier hashes", () => {
  it("matches the Python ordered UTF-8 frontier fixture", () => {
    expect(nativeAppObservationFrontierSha256(head)).toBe("e813400ad7d41e20608351a49e89dbade292bb54dd663451731097e6daf3257e");
    expect(nativeAppObservationFrontierSha256(null)).toBe(EMPTY_NATIVE_APP_OBSERVATION_FRONTIER);
  });
  it("does not normalize reviewed JSON bytes or source state", () => {
    expect(nativeAppObservationRawSha256('{"a":1}')).not.toBe(nativeAppObservationRawSha256('{ "a":1 }'));
    expect(nativeAppObservationFrontierSha256({ ...head, disposition: "superseded",
      state: "advanced-without-observation", learning: null })).not.toBe(nativeAppObservationFrontierSha256(head));
    expect(nativeAppObservationFrontierSha256({ ...head, learning: { ...head.learning!, afterimageId: "scratch-two" } }))
      .not.toBe(nativeAppObservationFrontierSha256(head));
  });
  it("rejects unpaired Unicode before cross-process hashing", () => {
    expect(() => nativeAppObservationRawSha256("\ud800")).toThrow(/UTF-8/);
    expect(() => nativeAppObservationRawSha256("\udc00")).toThrow(/UTF-8/);
  });
  it("matches the Python conditional content-head hash without changing text heads", () => {
    const hash = "d".repeat(64), order = "e".repeat(64);
    const mediaHead: NativeAppObservationHead = { ...head,
      learning: { ...head.learning!, textBytes: 0, textSha256: nativeAppObservationRawSha256("") },
      content: { summary: { format: "omni-native-app-content-summary", version: 1, blockCount: 1,
        textBlocks: 0, resourceTextBlocks: 0, referenceBlocks: 0, structuredBlocks: 0, imageBlocks: 1, audioBlocks: 0,
        decodedWindows: 1, imageFrames: 1, audioModelSamples: 0, sourceMediaBytes: 5,
        orderedInputSha256: order, wholeInputTraversed: true, nativeSensory: true },
      audit: { format: "omni-native-app-observation-audit", version: 1, path: `app-observation-audits/${hash}.jsonl`,
        sha256: hash, bytes: 500, rows: 1, sourceBrainId: "fixture-owner", recordSha256: "a".repeat(64),
        dataSha256: "b".repeat(64), orderedInputSha256: order } },
    };
    expect(nativeAppObservationFrontierSha256(mediaHead)).toBe("1fb2e499884681f6a454e8b99408c7e349d20ac928613d7e96156f920f85f099");
    expect(nativeAppObservationFrontierSha256(head)).toBe("e813400ad7d41e20608351a49e89dbade292bb54dd663451731097e6daf3257e");
  });
});
