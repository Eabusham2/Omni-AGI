import { createHash } from "node:crypto";
import type { ApprovedNativeMcpSamplingInput, NativeMcpSamplingControl, NativeMcpSamplingInput,
  NativeMcpSamplingResult, NativeMcpSamplingStatus, NativeMcpSamplingContent } from "../shared/nativeMcpSampling";
import { NATIVE_MCP_IMAGE_MIME_TYPES, NATIVE_MCP_AUDIO_MIME_TYPES } from "../shared/nativeMcpSampling";

const FORMAT = "omni-native-mcp-sampling-v1";
const ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const HASH = /^[a-f0-9]{64}$/;
const TERMINAL = new Set(["complete", "cancelled", "failed"]);

function text(value: unknown, identifier = false): asserts value is string {
  if (typeof value !== "string" || value.includes("\0") ||
      /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(value) ||
      identifier && !ID.test(value)) throw new Error("Invalid native MCP sampling text or ownership.");
}

export function nativeMcpSamplingMediaMetadata(block: NativeMcpSamplingContent): {
  kind: "image" | "audio"; mimeType: string; decodedBytes: number
} {
  if (!block || !["image", "audio"].includes(block.type) || Object.keys(block).sort().join(",") !== "data,mimeType,type") {
    throw new Error("Native MCP sampling media must be a typed inline image or audio block.");
  }
  const media = block as Extract<NativeMcpSamplingContent, { data: string }>;
  const accepted: readonly string[] = media.type === "image" ? NATIVE_MCP_IMAGE_MIME_TYPES : NATIVE_MCP_AUDIO_MIME_TYPES;
  if (typeof media.mimeType !== "string" || !accepted.includes(media.mimeType.toLowerCase())) {
    throw new Error(`Unsupported native MCP ${media.type} MIME type.`);
  }
  if (typeof media.data !== "string" || !media.data || media.data.length % 4 !== 0 ||
      !/^[A-Za-z0-9+/]*={0,2}$/.test(media.data) ||
      Buffer.from(media.data.slice(-4), "base64").toString("base64") !== media.data.slice(-4)) {
    throw new Error("Native MCP sampling media must contain canonical inline base64 data.");
  }
  const padding = media.data.endsWith("==") ? 2 : media.data.endsWith("=") ? 1 : 0;
  return { kind: media.type, mimeType: media.mimeType, decodedBytes: media.data.length / 4 * 3 - padding };
}

/** Hash decoded file bytes in bounded chunks; no image/audio decoder runs. */
export function nativeMcpSamplingMediaSourceSha256(block: NativeMcpSamplingContent): string {
  nativeMcpSamplingMediaMetadata(block);
  const media = block as Extract<NativeMcpSamplingContent, { data: string }>;
  const digest = createHash("sha256");
  for (let offset = 0; offset < media.data.length; offset += 64 * 1024) {
    digest.update(Buffer.from(media.data.slice(offset, offset + 64 * 1024), "base64"));
  }
  return digest.digest("hex");
}

function content(block: NativeMcpSamplingContent): NativeMcpSamplingContent {
  if (block?.type === "text" && Object.keys(block).sort().join(",") === "text,type") {
    text(block.text);
    return { type: "text", text: block.text };
  }
  nativeMcpSamplingMediaMetadata(block);
  const media = block as Extract<NativeMcpSamplingContent, { data: string }>;
  return { type: media.type, data: media.data, mimeType: media.mimeType };
}

/** The exact byte string hashed at review and authenticated by the worker.
 * Explicit field ordering avoids cross-language floating JSON conversions.
 */
export function nativeMcpSamplingInputJson(input: NativeMcpSamplingInput): string {
  const allowed = new Set(["brainId", "requestId", "originatingTurnId", "messages", "maxTokens", "temperature", "stopSequences", "inputSha256"]);
  if (!input || typeof input !== "object" || Object.keys(input).some((key) => !allowed.has(key))) {
    throw new Error("Native MCP sampling contains unreviewed fields.");
  }
  text(input.brainId, true); text(input.requestId, true);
  if (input.originatingTurnId !== undefined) text(input.originatingTurnId, true);
  if (!Number.isSafeInteger(input.maxTokens) || input.maxTokens < 1) throw new Error("Native MCP maxTokens must be a positive safe integer.");
  if (input.temperature !== undefined && (typeof input.temperature !== "number" || !Number.isFinite(input.temperature) || input.temperature < 0)) {
    throw new Error("Native MCP temperature must be finite and nonnegative.");
  }
  if (!Array.isArray(input.messages) || !input.messages.length) throw new Error("Native MCP sampling needs approved messages.");
  const messages = input.messages.map((message) => {
    if (!message || Object.keys(message).sort().join(",") !== "content,role" ||
        !["user", "assistant"].includes(message.role) || !message.content) {
      throw new Error("Native MCP sampling requires typed reviewed message blocks.");
    }
    if (Array.isArray(message.content) && !message.content.length) throw new Error("Native MCP sampling message blocks cannot be empty.");
    return { role: message.role, content: Array.isArray(message.content) ? message.content.map(content) : content(message.content) };
  });
  if (!messages.some((message) => (Array.isArray(message.content) ? message.content : [message.content]).some(
    (block) => block.type !== "text" || block.text.trim()))) throw new Error("Empty approval cannot enqueue native sampling.");
  if (input.stopSequences !== undefined && (!Array.isArray(input.stopSequences) || input.stopSequences.some((stop) => {
    text(stop); return !stop;
  }))) throw new Error("Native MCP stop sequences must be nonempty text.");
  return JSON.stringify({ format: FORMAT, brainId: input.brainId, requestId: input.requestId,
    ...(input.originatingTurnId !== undefined ? { originatingTurnId: input.originatingTurnId } : {}), messages,
    maxTokens: input.maxTokens, ...(input.temperature !== undefined ? { temperature: input.temperature } : {}),
    ...(input.stopSequences !== undefined ? { stopSequences: [...input.stopSequences] } : {}) });
}

export function nativeMcpSamplingInputSha256(input: NativeMcpSamplingInput): string {
  return createHash("sha256").update(nativeMcpSamplingInputJson(input), "utf8").digest("hex");
}

interface Owner { brainId: string; requestId: string; inputSha256: string }
export interface NativeMcpSamplingTransport {
  control(method: NativeMcpSamplingControl, params: Record<string, unknown>): Promise<unknown>;
  /** Serialized native sampling, not brain.chat, with no destructive abort. */
  runIdle(owner: Owner): Promise<unknown>;
}

function status(value: unknown, owner: Owner): NativeMcpSamplingStatus {
  const row = value as NativeMcpSamplingStatus | undefined;
  if (!row || row.brainId !== owner.brainId || row.requestId !== owner.requestId || row.inputSha256 !== owner.inputSha256 ||
      !["queued", "running", "complete", "cancelled", "failed"].includes(row.state)) {
    throw new Error("Native MCP sampling returned a different request owner.");
  }
  return row;
}

function result(value: NativeMcpSamplingStatus, owner: Owner, input: NativeMcpSamplingInput): NativeMcpSamplingResult {
  if (value.state !== "complete") throw new Error(value.error?.message || `Native MCP sampling ${value.state}.`);
  const row = value.result;
  if (!row || row.brainId !== owner.brainId || row.requestId !== owner.requestId || row.inputSha256 !== owner.inputSha256 ||
      row.role !== "assistant" || row.content?.type !== "text" || typeof row.content.text !== "string" ||
      row.model !== `OmniCortex/${owner.brainId}` || !["endTurn", "maxTokens", "stopSequence"].includes(row.stopReason) ||
      row.privateWorkspace !== true || row.parametersMutated !== false) {
    throw new Error("Native MCP sampling lacks a same-brain private result receipt.");
  }
  text(row.content.text);
  const media = input.messages.flatMap((message, messageIndex) => (Array.isArray(message.content) ? message.content : [message.content])
    .flatMap((block, blockIndex) => block.type === "text" ? [] : [{ ...nativeMcpSamplingMediaMetadata(block),
      sourceSha256: nativeMcpSamplingMediaSourceSha256(block), messageIndex, blockIndex }]));
  if (media.length || row.mediaInput?.length) {
    if (!Array.isArray(row.mediaInput) || row.mediaInput.length !== media.length) throw new Error("Native MCP sampling lacks full sensory input receipts.");
    for (const [index, expected] of media.entries()) {
      const observed = row.mediaInput[index];
      if (!observed || observed.messageIndex !== expected.messageIndex || observed.blockIndex !== expected.blockIndex ||
          observed.kind !== expected.kind || observed.mimeType !== expected.mimeType || observed.sourceBytes !== expected.decodedBytes ||
          observed.sourceSha256 !== expected.sourceSha256 || observed.wholeInputTraversed !== true || observed.nativeSensory !== true ||
          !Number.isSafeInteger(observed.decodedWindows) || observed.decodedWindows < 1 ||
          expected.kind === "image" && (!Number.isSafeInteger(observed.sourceFrames) || Number(observed.sourceFrames) < 1) ||
          expected.kind === "audio" && (!Number.isSafeInteger(observed.modelSamples) || Number(observed.modelSamples) < 1 || observed.modelSampleRate !== 16000)) {
        throw new Error("Native MCP sampling returned incomplete or different sensory traversal.");
      }
    }
  }
  return row;
}

const pause = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 50));

/** Never puts an inline sampling completion behind the chat awaiting it. */
export async function performNativeMcpSampling(transport: NativeMcpSamplingTransport,
  approved: ApprovedNativeMcpSamplingInput, signal?: AbortSignal): Promise<NativeMcpSamplingResult> {
  const approvedInputJson = nativeMcpSamplingInputJson(approved);
  if (!HASH.test(approved.inputSha256) || createHash("sha256").update(approvedInputJson, "utf8").digest("hex") !== approved.inputSha256) {
    throw new Error("Native MCP sampling input differs from the approved request.");
  }
  signal?.throwIfAborted();
  const owner = { brainId: approved.brainId, requestId: approved.requestId, inputSha256: approved.inputSha256 };
  let owned = false, terminal = false, cancellation: Promise<void> | undefined, idle: Promise<unknown> | undefined;
  let idleFailure: unknown;
  const cancel = (): void => {
    if (owned && !cancellation) cancellation = transport.control("mcp_sampling_cancel", owner).then(() => undefined);
    void cancellation?.catch(() => undefined);
  };
  signal?.addEventListener("abort", cancel, { once: true });
  try {
    let current = status(await transport.control("mcp_sampling_enqueue", { ...owner, approvedInputJson }), owner);
    if (current.accepted !== true || !["turn", "serial"].includes(current.boundary ?? "")) {
      throw new Error("Native MCP sampling did not acquire its reviewed writer boundary.");
    }
    owned = true;
    if (signal?.aborted) cancel();
    if (current.boundary === "serial" && !TERMINAL.has(current.state)) {
      // No signal is passed to the serial RPC: cancelling sampling sets its
      // exact private flag, never kills a worker holding an unrelated brain.
      idle = transport.runIdle(owner);
      void idle.catch((error: unknown) => { idleFailure = error; });
    }
    while (!TERMINAL.has(current.state)) {
      if (signal?.aborted) cancel();
      await pause();
      if (idleFailure) throw idleFailure;
      current = status(await transport.control("mcp_sampling_status", owner), owner);
    }
    terminal = true;
    if (idle && current.state === "complete") {
      const dispatched = status(await idle, owner);
      if (dispatched.state !== current.state) throw new Error("Native sampling serial/status receipts disagree.");
    }
    if (cancellation) await cancellation;
    signal?.throwIfAborted();
    return result(current, owner, approved);
  } finally {
    signal?.removeEventListener("abort", cancel);
    if (owned) {
      if (!terminal) {
        // A control failure is not permission to restart or replay the brain.
        // The caller receives the failure; the exact private owner is asked
        // to stop and its terminal receipt is consumed when reachable.
        await transport.control("mcp_sampling_cancel", owner).catch(() => undefined);
      }
      const last = await transport.control("mcp_sampling_status", owner).catch(() => undefined);
      if (last && TERMINAL.has(status(last, owner).state)) await transport.control("mcp_sampling_consume", owner);
    }
  }
}
