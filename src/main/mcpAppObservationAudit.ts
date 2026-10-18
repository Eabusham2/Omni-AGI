import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open } from "node:fs/promises";
import { join } from "node:path";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import type { NativeAppObservationAudit, NativeAppObservationBlockReceipt, NativeAppObservationContentSummary, NativeAppObservationHead } from "../shared/nativeAppObservation";
import { nativeAppObservationOrderedInputSha256, nativeAppObservationRawSha256 } from "./nativeAppObservation";
import { mcpAppTaskData } from "./mcpAppTaskData";
import { nativeMcpSamplingMediaMetadata, nativeMcpSamplingMediaSourceSha256 } from "./nativeMcpSampling";
import { NATIVE_MCP_AUDIO_MIME_TYPES, NATIVE_MCP_IMAGE_MIME_TYPES } from "../shared/nativeMcpSampling";

const SHA = /^[a-f0-9]{64}$/, ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const object = (value: unknown): Record<string, unknown> => value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const nonnegative = (value: unknown): value is number => Number.isSafeInteger(value) && Number(value) >= 0;
export function safeAppObservationAuditPath(path: string): boolean { return /^app-observation-audits\/[a-f0-9]{64}\.jsonl$/.test(path); }

export function checkedAppObservationContent(value: unknown, recordSha256: string, dataSha256: string): NonNullable<NativeAppObservationHead["content"]> {
  const content = object(value), s = object(content.summary), a = object(content.audit);
  if (Object.keys(content).sort().join(",") !== "audit,summary" ||
    Object.keys(s).sort().join(",") !== "audioBlocks,audioModelSamples,blockCount,decodedWindows,format,imageBlocks,imageFrames,nativeSensory,orderedInputSha256,referenceBlocks,resourceTextBlocks,sourceMediaBytes,structuredBlocks,textBlocks,version,wholeInputTraversed" ||
    s.format !== "omni-native-app-content-summary" || s.version !== 1 || s.wholeInputTraversed !== true || typeof s.nativeSensory !== "boolean" || !SHA.test(String(s.orderedInputSha256)) ||
    ![s.blockCount, s.textBlocks, s.resourceTextBlocks, s.referenceBlocks, s.structuredBlocks, s.imageBlocks, s.audioBlocks, s.decodedWindows, s.imageFrames, s.audioModelSamples, s.sourceMediaBytes].every(nonnegative) || Number(s.blockCount) < 1 ||
    s.blockCount !== Number(s.textBlocks) + Number(s.resourceTextBlocks) + Number(s.referenceBlocks) + Number(s.structuredBlocks) + Number(s.imageBlocks) + Number(s.audioBlocks) ||
    s.nativeSensory !== (Number(s.imageBlocks) + Number(s.audioBlocks) > 0) ||
    Number(s.structuredBlocks) > 1 || Number(s.decodedWindows) < Number(s.imageBlocks) + Number(s.audioBlocks) || Number(s.imageFrames) < Number(s.imageBlocks) ||
    Number(s.imageFrames) > Number(s.decodedWindows) || Number(s.audioModelSamples) < Number(s.audioBlocks) || s.nativeSensory && Number(s.sourceMediaBytes) < 1 ||
    !s.nativeSensory && (s.decodedWindows !== 0 || s.imageFrames !== 0 || s.audioModelSamples !== 0 || s.sourceMediaBytes !== 0)) throw new Error("App sensory summary lacks truthful complete ordered coverage.");
  if (Object.keys(a).sort().join(",") !== "bytes,dataSha256,format,orderedInputSha256,path,recordSha256,rows,sha256,sourceBrainId,version" ||
    a.format !== "omni-native-app-observation-audit" || a.version !== 1 || typeof a.sourceBrainId !== "string" || !ID.test(a.sourceBrainId) || !SHA.test(String(a.sha256)) ||
    typeof a.path !== "string" || !safeAppObservationAuditPath(a.path) || a.path !== `app-observation-audits/${a.sha256}.jsonl` ||
    !nonnegative(a.bytes) || a.bytes < 1 || a.rows !== s.blockCount || a.recordSha256 !== recordSha256 || a.dataSha256 !== dataSha256 ||
    a.orderedInputSha256 !== s.orderedInputSha256) throw new Error("App sensory audit has an unsafe path or changed source binding.");
  return content as unknown as NonNullable<NativeAppObservationHead["content"]>;
}
interface AuditHooks { signal?: AbortSignal; admit?(bytes: number): Promise<void> }
interface ExpectedAuditBlock { kind: NativeAppObservationBlockReceipt["kind"]; sourceEncoding: NativeAppObservationBlockReceipt["sourceEncoding"]; sourceSha256: string; sourceBytes: number; media?: { type: "image" | "audio"; data: string; mimeType: string } }
function expectedMedia(media: NonNullable<ExpectedAuditBlock["media"]>): ExpectedAuditBlock {
  const metadata = nativeMcpSamplingMediaMetadata(media);
  return { kind: media.type, sourceEncoding: "decoded-inline-media", sourceSha256: nativeMcpSamplingMediaSourceSha256(media), sourceBytes: metadata.decodedBytes, media };
}
async function admit(hooks: AuditHooks, bytes: number): Promise<void> {
  hooks.signal?.throwIfAborted();
  if (hooks.admit) return hooks.admit(bytes);
  if (!Number.isSafeInteger(bytes) || bytes > freemem() || bytes > getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) throw new Error("App audit verification awaits physical transfer RAM.");
}

/** Verify receipts as measurements of the approved data, not as an answer store.
 * No image/audio decoder, linked-resource fetch or model call runs here. */
export async function verifyAppObservationAudit(path: string, content: NonNullable<NativeAppObservationHead["content"]>, dataJson?: string, hooks: AuditHooks = {}): Promise<void> {
  const { summary, audit } = checkedAppObservationContent(content, content.audit.recordSha256, content.audit.dataSha256);
  const before = await lstat(path); if (!before.isFile() || before.isSymbolicLink() || before.size !== audit.bytes) throw new Error("App audit is not its exact owned regular source file.");
  let expected: ExpectedAuditBlock[] | undefined;
  if (dataJson !== undefined) {
    await admit(hooks, Buffer.byteLength(dataJson) * 16 + 65536);
    if (nativeAppObservationRawSha256(dataJson) !== audit.dataSha256) throw new Error("App audit differs from the exact approved data bytes.");
    const parsed = JSON.parse(dataJson), data = mcpAppTaskData(parsed.method, parsed.params), content = data.params.content;
    const blocks = Array.isArray(content) ? content : content ? [content] : [], wholeBytes = Buffer.byteLength(dataJson);
    expected = blocks.map((block): ExpectedAuditBlock => {
      if (block.type === "text") return { kind: "text", sourceEncoding: "utf8-text", sourceSha256: nativeAppObservationRawSha256(block.text), sourceBytes: Buffer.byteLength(block.text) };
      if (block.type === "resource" && typeof block.resource.text === "string") return { kind: "resource-text", sourceEncoding: "utf8-text", sourceSha256: nativeAppObservationRawSha256(block.resource.text), sourceBytes: Buffer.byteLength(block.resource.text) };
      if (block.type === "image" || block.type === "audio") {
        return expectedMedia({ type: block.type, data: block.data, mimeType: block.mimeType });
      }
      if (block.type === "resource" && typeof block.resource.blob === "string" && typeof block.resource.mimeType === "string") {
        const mime = block.resource.mimeType.toLowerCase();
        const type = (NATIVE_MCP_IMAGE_MIME_TYPES as readonly string[]).includes(mime) ? "image" :
          (NATIVE_MCP_AUDIO_MIME_TYPES as readonly string[]).includes(mime) ? "audio" : undefined;
        if (!type) throw new Error("App ordinary learning does not invent a decoder for unsupported binary resources.");
        return expectedMedia({ type, data: block.resource.blob, mimeType: block.resource.mimeType });
      }
      if (block.type !== "resource_link") throw new Error("App ordinary learning does not treat an embedded binary resource as text.");
      return { kind: "resource-reference", sourceEncoding: "approved-data-json", sourceSha256: audit.dataSha256, sourceBytes: wholeBytes };
    });
    if (data.method === "ui/update-model-context" && data.params.structuredContent !== undefined) expected.push({ kind: "structured", sourceEncoding: "approved-data-json", sourceSha256: audit.dataSha256, sourceBytes: wholeBytes });
    if (expected.length !== audit.rows) throw new Error("App audit did not visit every approved content/structured block.");
  }
  const count = { blockCount: 0, textBlocks: 0, resourceTextBlocks: 0, referenceBlocks: 0, structuredBlocks: 0, imageBlocks: 0, audioBlocks: 0, decodedWindows: 0, imageFrames: 0, audioModelSamples: 0, sourceMediaBytes: 0 };
  let ordered = "0".repeat(64), bytes = 0, pending = "";
  const hash = createHash("sha256"), decoder = new TextDecoder("utf-8", { fatal: true });
  const row = async (line: string): Promise<void> => {
    if (count.blockCount >= audit.rows) throw new Error("App audit contains more blocks than its approved declaration.");
    await admit(hooks, Buffer.byteLength(line) * 16 + 4096); const value = object(JSON.parse(line));
    const keys = Object.keys(value).sort().join(","), media = value.media === undefined ? undefined : object(value.media), source = expected?.[count.blockCount];
    if (keys !== "blockIndex,dataSha256,fetched,format,kind,nativeSensory,recordSha256,sourceBrainId,sourceBytes,sourceEncoding,sourceSha256,version,wholeInputTraversed" &&
      keys !== "blockIndex,dataSha256,fetched,format,kind,media,nativeSensory,recordSha256,sourceBrainId,sourceBytes,sourceEncoding,sourceSha256,version,wholeInputTraversed") throw new Error("App audit row contains an unsupported field.");
    if (value.format !== "omni-native-app-observation-block" || value.version !== 1 || value.sourceBrainId !== audit.sourceBrainId || value.recordSha256 !== audit.recordSha256 || value.dataSha256 !== audit.dataSha256 ||
      value.blockIndex !== count.blockCount || !["text", "resource-text", "resource-reference", "structured", "image", "audio"].includes(String(value.kind)) ||
      !["utf8-text", "approved-data-json", "decoded-inline-media"].includes(String(value.sourceEncoding)) || !SHA.test(String(value.sourceSha256)) || !nonnegative(value.sourceBytes) ||
      value.wholeInputTraversed !== true || value.fetched !== false || value.nativeSensory !== ["image", "audio"].includes(String(value.kind))) throw new Error("App audit row changed its source/order/coverage.");
    const encoding = ["text", "resource-text"].includes(String(value.kind)) ? "utf8-text" : value.nativeSensory ? "decoded-inline-media" : "approved-data-json";
    if (value.sourceEncoding !== encoding) throw new Error("App audit changed the declared source encoding.");
    if (source && (source.kind !== value.kind || source.sourceEncoding !== value.sourceEncoding || source.sourceSha256 !== value.sourceSha256 || source.sourceBytes !== value.sourceBytes)) throw new Error("App audit measurement does not match its actual approved source block.");
    if (value.nativeSensory) {
      if (media && Object.keys(media).some(key => !["messageIndex", "blockIndex", "kind", "mimeType", "sourceBytes", "sourceSha256", "decodedWindows", "sourceFrames", "sourceSampleFrames", "sourceSampleRate", "modelSamples", "modelSampleRate", "wholeInputTraversed", "nativeSensory"].includes(key))) throw new Error("App media audit contains an unsupported measurement field.");
      if (!media || media.messageIndex !== 0 || media.blockIndex !== value.blockIndex || media.kind !== value.kind || media.sourceSha256 !== value.sourceSha256 || media.sourceBytes !== value.sourceBytes ||
        typeof media.mimeType !== "string" || source?.media && media.mimeType !== source.media.mimeType || media.nativeSensory !== true || media.wholeInputTraversed !== true || !nonnegative(media.decodedWindows) || media.decodedWindows < 1) throw new Error("App media receipt did not traverse its real sensory source.");
      count.decodedWindows += media.decodedWindows; count.sourceMediaBytes += Number(media.sourceBytes);
      if (value.kind === "image") { if (!nonnegative(media.sourceFrames) || media.sourceFrames !== media.decodedWindows) throw new Error("App image receipt omitted decoded frames."); count.imageFrames += media.sourceFrames; }
      else { if (!nonnegative(media.modelSamples) || media.modelSamples < 1 || media.modelSampleRate !== 16000 || !nonnegative(media.sourceSampleFrames) || media.sourceSampleFrames < 1 || !nonnegative(media.sourceSampleRate) || media.sourceSampleRate < 1) throw new Error("App audio receipt omitted actual sample coverage."); count.audioModelSamples += media.modelSamples; }
    } else if (media) throw new Error("Text/reference App data cannot fabricate a sensory receipt.");
    const field = { text: "textBlocks", "resource-text": "resourceTextBlocks", "resource-reference": "referenceBlocks", structured: "structuredBlocks", image: "imageBlocks", audio: "audioBlocks" }[String(value.kind)] as keyof typeof count;
    count[field]++; ordered = nativeAppObservationOrderedInputSha256(ordered, value as unknown as NativeAppObservationBlockReceipt); count.blockCount++;
    if (!Object.values(count).every(nonnegative)) throw new Error("App audit coverage counters exceed exact integer storage.");
  };
  const handle = await open(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
  try {
    const opened = await handle.stat();
    if (!opened.isFile() || opened.ino !== before.ino || opened.dev !== before.dev || opened.size !== before.size || opened.mtimeMs !== before.mtimeMs) throw new Error("App audit source was replaced before it could be opened.");
    for await (const chunk of handle.createReadStream({ highWaterMark: 65536, signal: hooks.signal, autoClose: false })) {
      const data = chunk as Buffer; bytes += data.length; if (bytes > audit.bytes) throw new Error("App audit grew during verification."); hash.update(data); pending += decoder.decode(data, { stream: true });
      await admit(hooks, Buffer.byteLength(pending) * 16 + 65536);
      let newline = pending.indexOf("\n"); while (newline >= 0) { await row(pending.slice(0, newline)); pending = pending.slice(newline + 1); newline = pending.indexOf("\n"); }
    }
    pending += decoder.decode(); if (pending) await row(pending);
    const after = await lstat(path), final = await handle.stat();
    if (after.isSymbolicLink() || !after.isFile() || [after, final].some(info => info.ino !== before.ino || info.dev !== before.dev || info.size !== before.size || info.mtimeMs !== before.mtimeMs) || bytes !== audit.bytes || hash.digest("hex") !== audit.sha256 ||
      count.blockCount !== audit.rows || ordered !== audit.orderedInputSha256 || Object.entries(count).some(([key, value]) => value !== summary[key as keyof NativeAppObservationContentSummary])) throw new Error("App sensory audit checksum, ordered coverage or summary changed.");
  } finally { await handle.close(); }
}

export async function savedAppObservationAuditFiles(metadata: unknown, engine: string, hooks: AuditHooks = {}): Promise<Map<string, string>> {
  const state = object(metadata), head = object(state.app_observation_frontier), result = new Map<string, string>();
  if (head.content === undefined) return result;
  const content = checkedAppObservationContent(head.content, String(head.recordSha256), String(head.dataSha256));
  for (const directory of [engine, join(engine, "app-observation-audits")]) {
    const info = await lstat(directory);
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("App audit directory is not owned regular storage.");
  }
  const path = join(engine, ...content.audit.path.split("/")); await verifyAppObservationAudit(path, content, undefined, hooks);
  result.set(content.audit.path, path); return result;
}
