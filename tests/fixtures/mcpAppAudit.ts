/** Synthetic host receipt fixture. This does not run or prove a media decoder. */
import type { NativeAppObservationBlockReceipt, NativeAppObservationHead } from "../../src/shared/nativeAppObservation";
import { nativeAppObservationOrderedInputSha256, nativeAppObservationRawSha256 } from "../../src/main/nativeAppObservation";
import { nativeMcpSamplingMediaMetadata, nativeMcpSamplingMediaSourceSha256 } from "../../src/main/nativeMcpSampling";
import { mcpAppTaskData } from "../../src/main/mcpAppTaskData";
import { NATIVE_MCP_AUDIO_MIME_TYPES, NATIVE_MCP_IMAGE_MIME_TYPES } from "../../src/shared/nativeMcpSampling";

export function syntheticAppAudit(dataJson: string, recordSha256: string, sourceBrainId: string): {
  rows: NativeAppObservationBlockReceipt[]; raw: string; content: NonNullable<NativeAppObservationHead["content"]>
} {
  const parsed = JSON.parse(dataJson), data = mcpAppTaskData(parsed.method, parsed.params), body = data.params.content;
  const blocks = Array.isArray(body) ? body : body ? [body] : [], dataSha256 = nativeAppObservationRawSha256(dataJson);
  const base = { format: "omni-native-app-observation-block" as const, version: 1 as const, sourceBrainId,
    recordSha256, dataSha256, wholeInputTraversed: true as const, fetched: false as const };
  const rows = blocks.map((block, blockIndex): NativeAppObservationBlockReceipt => {
    if (block.type === "text" || block.type === "resource" && typeof block.resource.text === "string") {
      const text = block.type === "text" ? block.text : block.resource.text!;
      return { ...base, blockIndex, kind: block.type === "text" ? "text" : "resource-text", sourceEncoding: "utf8-text",
        sourceSha256: nativeAppObservationRawSha256(text), sourceBytes: Buffer.byteLength(text), nativeSensory: false };
    }
    const mime = block.type === "resource" ? block.resource.mimeType : undefined;
    const alias = typeof mime === "string" && block.type === "resource" && typeof block.resource.blob === "string" ? {
      type: (NATIVE_MCP_IMAGE_MIME_TYPES as readonly string[]).includes(mime.toLowerCase()) ? "image" as const :
        (NATIVE_MCP_AUDIO_MIME_TYPES as readonly string[]).includes(mime.toLowerCase()) ? "audio" as const : undefined,
      data: block.resource.blob, mimeType: mime } : undefined;
    const input = block.type === "image" || block.type === "audio" ? { type: block.type, data: block.data, mimeType: block.mimeType } :
      alias?.type ? { ...alias, type: alias.type } : undefined;
    if (input) {
      const info = nativeMcpSamplingMediaMetadata(input);
      const sourceSha256 = nativeMcpSamplingMediaSourceSha256(input), media = { messageIndex: 0, blockIndex, kind: input.type,
        mimeType: input.mimeType, sourceBytes: info.decodedBytes, sourceSha256, decodedWindows: 1, wholeInputTraversed: true as const,
        nativeSensory: true as const, ...(input.type === "image" ? { sourceFrames: 1 } : {
          sourceSampleFrames: 4, sourceSampleRate: 16000, modelSamples: 4, modelSampleRate: 16000 }) };
      return { ...base, blockIndex, kind: input.type, sourceEncoding: "decoded-inline-media", sourceSha256,
        sourceBytes: info.decodedBytes, nativeSensory: true, media };
    }
    if (block.type !== "resource_link") throw new Error("This fixture does not invent binary-resource decoding.");
    return { ...base, blockIndex, kind: "resource-reference", sourceEncoding: "approved-data-json", sourceSha256: dataSha256,
      sourceBytes: Buffer.byteLength(dataJson), nativeSensory: false };
  });
  if (data.method === "ui/update-model-context" && data.params.structuredContent !== undefined) rows.push({ ...base, blockIndex: rows.length,
    kind: "structured", sourceEncoding: "approved-data-json", sourceSha256: dataSha256, sourceBytes: Buffer.byteLength(dataJson), nativeSensory: false });
  return syntheticAppAuditRows(rows);
}

/** Re-seals deliberately tampered measurements for negative source-binding tests. */
export function syntheticAppAuditRows(rows: NativeAppObservationBlockReceipt[]): ReturnType<typeof syntheticAppAudit> {
  const first = rows[0]; if (!first) throw new Error("Synthetic audit needs a block.");
  const raw = rows.map(row => JSON.stringify(row) + "\n").join(""), sha256 = nativeAppObservationRawSha256(raw);
  const summary: NonNullable<NativeAppObservationHead["content"]>["summary"] = {
    format: "omni-native-app-content-summary", version: 1, blockCount: rows.length,
    textBlocks: 0, resourceTextBlocks: 0, referenceBlocks: 0, structuredBlocks: 0, imageBlocks: 0, audioBlocks: 0,
    decodedWindows: 0, imageFrames: 0, audioModelSamples: 0, sourceMediaBytes: 0,
    orderedInputSha256: "0".repeat(64), wholeInputTraversed: true, nativeSensory: false
  };
  const fields = { text: "textBlocks", "resource-text": "resourceTextBlocks", "resource-reference": "referenceBlocks",
    structured: "structuredBlocks", image: "imageBlocks", audio: "audioBlocks" } as const;
  for (const row of rows) {
    summary[fields[row.kind]]++; summary.orderedInputSha256 = nativeAppObservationOrderedInputSha256(summary.orderedInputSha256, row);
    if (row.media) { summary.nativeSensory = true; summary.decodedWindows += row.media.decodedWindows;
      summary.sourceMediaBytes += row.sourceBytes; summary.imageFrames += row.media.sourceFrames ?? 0;
      summary.audioModelSamples += row.media.modelSamples ?? 0; }
  }
  return { rows, raw, content: { summary, audit: { format: "omni-native-app-observation-audit", version: 1,
    path: `app-observation-audits/${sha256}.jsonl`, sha256, bytes: Buffer.byteLength(raw), rows: rows.length,
    sourceBrainId: first.sourceBrainId, recordSha256: first.recordSha256, dataSha256: first.dataSha256, orderedInputSha256: summary.orderedInputSha256 } } };
}

export const APP_AUDIT_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/lXYAAAAASUVORK5CYII=";
const wav = Buffer.alloc(52); wav.write("RIFF"); wav.writeUInt32LE(44, 4); wav.write("WAVEfmt ", 8);
wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22); wav.writeUInt32LE(16000, 24);
wav.writeUInt32LE(32000, 28); wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34); wav.write("data", 36); wav.writeUInt32LE(8, 40);
export const APP_AUDIT_WAV = wav.toString("base64");
