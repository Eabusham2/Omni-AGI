import { createHash } from "node:crypto";
import type { NativeAppObservationHead } from "../shared/nativeAppObservation";
import { EMPTY_NATIVE_APP_OBSERVATION_FRONTIER } from "../shared/nativeAppObservation";

/** Hash reviewed byte strings exactly; no semantic JSON normalization. */
export function nativeAppObservationRawSha256(raw: string): string {
  if (typeof raw !== "string" || /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(raw)) {
    throw new Error("App observation must contain valid UTF-8 text.");
  }
  return createHash("sha256").update(raw, "utf8").digest("hex");
}

/** Fixed ordered fields avoid cross-language object-order or number formatting. */
export function nativeAppObservationFrontierSha256(head: NativeAppObservationHead | null): string {
  if (head === null) return EMPTY_NATIVE_APP_OBSERVATION_FRONTIER;
  const learning = head.learning;
  const values: unknown[] = [
    head.format, head.version, head.streamId, head.ordinal, head.recordSha256,
    head.dataSha256, head.previousSha256, head.baseFrontierSha256,
    head.disposition, head.state, head.committedAt,
    learning === null ? null : [learning.kind, learning.source, learning.sourceLabel,
      learning.steps, learning.experienceCountBefore, learning.experienceCountAfter,
      learning.assemblyId, learning.afterimageId, learning.textSha256, learning.textBytes],
  ];
  if (head.content) {
    const { summary: s, audit: a } = head.content;
    values.push([[s.format, s.version, s.blockCount, s.textBlocks, s.resourceTextBlocks, s.referenceBlocks,
      s.structuredBlocks, s.imageBlocks, s.audioBlocks, s.decodedWindows, s.imageFrames, s.audioModelSamples,
      s.sourceMediaBytes, s.orderedInputSha256, s.wholeInputTraversed, s.nativeSensory],
      [a.format, a.version, a.path, a.sha256, a.bytes, a.rows, a.sourceBrainId,
        a.recordSha256, a.dataSha256, a.orderedInputSha256]]);
  }
  return nativeAppObservationRawSha256(JSON.stringify(values));
}

export function nativeAppObservationOrderedInputSha256(previous: string,
  block: import("../shared/nativeAppObservation").NativeAppObservationBlockReceipt): string {
  if (!/^[a-f0-9]{64}$/.test(previous)) throw new Error("Invalid App input order predecessor.");
  return nativeAppObservationRawSha256(previous + "\0" + JSON.stringify([
    block.blockIndex, block.kind, block.sourceEncoding, block.sourceSha256, block.sourceBytes,
  ]));
}
