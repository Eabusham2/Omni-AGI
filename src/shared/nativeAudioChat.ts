/** A real audio turn, never a platform transcript or a generated caption. */
export interface NativeAudioChatRequest {
  format: "omni-native-audio-chat";
  version: 1;
  captureSessionId: string;
  utteranceId: string;
  sequence: number;
  mimeType: string;
  dataBase64: string;
  capturedAt: string;
  sampleRate?: number;
  channels?: number;
}

/** Main-owned resolved attachment. No renderer pathname or permission grant. */
export interface NativeAudioAttachment {
  brainId: string;
  turnId: string;
  attachmentId: string;
  contentSha256: string;
  bytes: number;
  mimeType: string;
  kind: string;
  relativePath: string;
  displayName: string;
  manifestSha256: string;
}

export interface NativeAudioInput {
  format: "omni-native-audio-input";
  version: 1;
  attachment: NativeAudioAttachment;
  capture: {
    sessionId: string;
    utteranceId: string;
    sequence: number;
    capturedAt: string;
    permissionRevision: number;
  };
  inputSha256: string;
}

/** Durable source attribution. This is never turned into model prompt text. */
export interface NativeAudioInputReceipt {
  input: NativeAudioInput;
  assemblyId: string;
  sourceBytes: number;
  sourceSha256: string;
  decodedWindows: number;
  modelSamples: number;
  modelSampleRate: number;
  sourceSampleFrames?: number;
  sourceSampleRate?: number;
  wholeInputTraversed: true;
  nativeSensory: true;
  transcriptUsed: false;
}

/** Renderer-safe measured input. No filesystem path or reusable permission. */
export interface NativeAudioMessageInfo {
  kind: "native-audio";
  inputSha256: string;
  attachmentId: string;
  sourceBrainId: string;
  sourceTurnId: string;
  utteranceId: string;
  sequence: number;
  capturedAt: string;
  mimeType: string;
  assemblyId: string;
  sourceBytes: number;
  sourceSha256: string;
  decodedWindows: number;
  modelSamples: number;
  modelSampleRate: number;
  wholeInputTraversed: true;
  nativeSensory: true;
  transcriptUsed: false;
}

export function nativeAudioMessageInfo(receipt: NativeAudioInputReceipt): NativeAudioMessageInfo {
  const { input } = receipt, source = input.attachment;
  return { kind: "native-audio", inputSha256: input.inputSha256, attachmentId: source.attachmentId,
    sourceBrainId: source.brainId, sourceTurnId: source.turnId, utteranceId: input.capture.utteranceId,
    sequence: input.capture.sequence, capturedAt: input.capture.capturedAt, mimeType: source.mimeType,
    assemblyId: receipt.assemblyId, sourceBytes: receipt.sourceBytes, sourceSha256: receipt.sourceSha256,
    decodedWindows: receipt.decodedWindows, modelSamples: receipt.modelSamples, modelSampleRate: receipt.modelSampleRate,
    wholeInputTraversed: true, nativeSensory: true, transcriptUsed: false };
}

export function checkedNativeAudioInfo(value: unknown, expected?: NativeAudioInput): NativeAudioMessageInfo {
  const row = object(value);
  exact(row, ["kind", "inputSha256", "attachmentId", "sourceBrainId", "sourceTurnId", "utteranceId", "sequence", "capturedAt", "mimeType", "assemblyId", "sourceBytes", "sourceSha256", "decodedWindows", "modelSamples", "modelSampleRate", "wholeInputTraversed", "nativeSensory", "transcriptUsed"]);
  if (row.kind !== "native-audio" || !hash(row.inputSha256) || !hash(row.sourceSha256) ||
      ![row.attachmentId, row.sourceBrainId, row.sourceTurnId, row.utteranceId, row.assemblyId].every(id) ||
      !integer(row.sequence) || ![row.sourceBytes, row.decodedWindows, row.modelSamples, row.modelSampleRate].every(value => integer(value, 1)) ||
      typeof row.capturedAt !== "string" || !Number.isFinite(Date.parse(row.capturedAt)) || typeof row.mimeType !== "string" || !AUDIO.has(row.mimeType) ||
      row.wholeInputTraversed !== true || row.nativeSensory !== true || row.transcriptUsed !== false) throw new Error("Native audio presentation is not a complete measured input.");
  if (expected && (row.inputSha256 !== expected.inputSha256 || row.attachmentId !== expected.attachment.attachmentId ||
      row.sourceBrainId !== expected.attachment.brainId || row.sourceTurnId !== expected.attachment.turnId ||
      row.sourceBytes !== expected.attachment.bytes || row.sourceSha256 !== expected.attachment.contentSha256 || row.mimeType !== expected.attachment.mimeType ||
      row.utteranceId !== expected.capture.utteranceId || row.sequence !== expected.capture.sequence || row.capturedAt !== expected.capture.capturedAt)) throw new Error("Native audio presentation belongs to another input.");
  return { ...row } as unknown as NativeAudioMessageInfo;
}

const ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/, SHA = /^[a-f0-9]{64}$/;
const AUDIO = new Set(["audio/wav", "audio/x-wav", "audio/wave", "audio/flac", "audio/x-flac",
  "audio/ogg", "audio/mpeg", "audio/mp3", "audio/aac", "audio/mp4", "audio/x-m4a"]);
const object = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Native audio input must be typed data.");
  return value as Record<string, unknown>;
};
const exact = (row: Record<string, unknown>, required: string[], optional: string[] = []): void => {
  if (required.some(key => !(key in row)) || Object.keys(row).some(key => !required.includes(key) && !optional.includes(key)))
    throw new Error("Native audio input contains unknown or missing fields.");
};
const id = (value: unknown): value is string => typeof value === "string" && ID.test(value);
const hash = (value: unknown): value is string => typeof value === "string" && SHA.test(value);
const integer = (value: unknown, min = 0): value is number => Number.isSafeInteger(value) && Number(value) >= min;

export function checkedNativeAudioRequest(value: unknown): NativeAudioChatRequest {
  const row = object(value);
  exact(row, ["format", "version", "captureSessionId", "utteranceId", "sequence", "mimeType", "dataBase64", "capturedAt"], ["sampleRate", "channels"]);
  if (row.format !== "omni-native-audio-chat" || row.version !== 1 || !id(row.captureSessionId) || !id(row.utteranceId) || !integer(row.sequence) ||
      typeof row.mimeType !== "string" || !AUDIO.has(row.mimeType) || typeof row.dataBase64 !== "string" || !row.dataBase64 ||
      row.dataBase64.length % 4 || !/^[A-Za-z0-9+/]+={0,2}$/.test(row.dataBase64) ||
      typeof row.capturedAt !== "string" || !Number.isFinite(Date.parse(row.capturedAt)) ||
      row.sampleRate !== undefined && !integer(row.sampleRate, 1) || row.channels !== undefined && !integer(row.channels, 1))
    throw new Error("Native audio capture identity, codec, or raw body is invalid.");
  return { ...row } as unknown as NativeAudioChatRequest;
}

export function checkedNativeAudioInput(value: unknown): NativeAudioInput {
  const row = object(value);
  exact(row, ["format", "version", "attachment", "capture", "inputSha256"]);
  const attachment = object(row.attachment), capture = object(row.capture);
  exact(attachment, ["brainId", "turnId", "attachmentId", "contentSha256", "bytes", "mimeType", "kind", "relativePath", "displayName", "manifestSha256"]);
  exact(capture, ["sessionId", "utteranceId", "sequence", "capturedAt", "permissionRevision"]);
  if (row.format !== "omni-native-audio-input" || row.version !== 1 || !hash(row.inputSha256) ||
      !id(attachment.brainId) || !id(attachment.turnId) || !id(attachment.attachmentId) || !hash(attachment.contentSha256) || !hash(attachment.manifestSha256) ||
      !integer(attachment.bytes, 1) || attachment.kind !== "audio" || typeof attachment.mimeType !== "string" || !AUDIO.has(attachment.mimeType) ||
      attachment.relativePath !== `attachments/blobs/${attachment.contentSha256}.data` ||
      typeof attachment.displayName !== "string" || /\0/.test(attachment.displayName) ||
      !id(capture.sessionId) || !id(capture.utteranceId) || !integer(capture.sequence) || !integer(capture.permissionRevision) ||
      typeof capture.capturedAt !== "string" || !Number.isFinite(Date.parse(capture.capturedAt))) throw new Error("Native audio attachment lost its exact source binding.");
  return { format: "omni-native-audio-input", version: 1,
    attachment: { brainId: attachment.brainId, turnId: attachment.turnId, attachmentId: attachment.attachmentId,
      contentSha256: attachment.contentSha256, bytes: Number(attachment.bytes), mimeType: attachment.mimeType,
      kind: "audio", relativePath: String(attachment.relativePath), displayName: attachment.displayName,
      manifestSha256: attachment.manifestSha256 },
    capture: { sessionId: capture.sessionId, utteranceId: capture.utteranceId, sequence: Number(capture.sequence),
      capturedAt: capture.capturedAt, permissionRevision: Number(capture.permissionRevision) }, inputSha256: row.inputSha256 };
}

/** Stable wire identity; the digest is computed by trusted main/native code. */
export function nativeAudioIdentityJson(input: Omit<NativeAudioInput, "inputSha256"> | NativeAudioInput): string {
  const canonical = (value: unknown): unknown => Array.isArray(value) ? value.map(canonical)
    : value !== null && typeof value === "object" ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, item]) => [key, canonical(item)])) : value;
  return JSON.stringify(canonical({ format: input.format, version: input.version, attachment: input.attachment, capture: input.capture }));
}

export function checkedNativeAudioReceipt(value: unknown): NativeAudioInputReceipt {
  const row = object(value);
  exact(row, ["input", "assemblyId", "sourceBytes", "sourceSha256", "decodedWindows", "modelSamples", "modelSampleRate", "wholeInputTraversed", "nativeSensory", "transcriptUsed"], ["sourceSampleFrames", "sourceSampleRate"]);
  const input = checkedNativeAudioInput(row.input);
  if (!id(row.assemblyId) || row.sourceBytes !== input.attachment.bytes || row.sourceSha256 !== input.attachment.contentSha256 ||
      !integer(row.decodedWindows, 1) || !integer(row.modelSamples, 1) || !integer(row.modelSampleRate, 1) ||
      row.sourceSampleFrames !== undefined && !integer(row.sourceSampleFrames, 1) || row.sourceSampleRate !== undefined && !integer(row.sourceSampleRate, 1) ||
      row.wholeInputTraversed !== true || row.nativeSensory !== true || row.transcriptUsed !== false) throw new Error("Native audio receipt does not prove complete actual sensory input.");
  return { ...row, input } as unknown as NativeAudioInputReceipt;
}
