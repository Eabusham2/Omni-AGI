import { createHash } from "node:crypto";
import { checkedNativeAudioRequest, nativeAudioIdentityJson, type NativeAudioInput } from "../shared/nativeAudioChat";
import type { TaskAttachmentService } from "./taskAttachmentService";
import { verifiedNativeAudioInput } from "./nativeAudioIdentity";

export interface NativeAudioCaptureLeaseRequest {
  brainId: string;
  turnId: string;
  sessionId: string;
  utteranceId: string;
  sequence: number;
  capturedAt: string;
  bytes: number;
  contentSha256: string;
}

export type NativeAudioCaptureLeaseCheck = (request: NativeAudioCaptureLeaseRequest) => { permissionRevision: number; maxBytes: number };

const sha = (value: string | Uint8Array): string => createHash("sha256").update(value).digest("hex");

/** Reuses the task attachment store; no second blob or permission registry. */
export async function prepareNativeAudioChat(
  attachments: TaskAttachmentService,
  requireLease: NativeAudioCaptureLeaseCheck,
  brainId: string,
  turnId: string,
  value: unknown,
  signal?: AbortSignal
): Promise<NativeAudioInput> {
  const request = checkedNativeAudioRequest(value);
  signal?.throwIfAborted();
  const padding = request.dataBase64.endsWith("==") ? 2 : request.dataBase64.endsWith("=") ? 1 : 0;
  const bytes = request.dataBase64.length / 4 * 3 - padding;
  // Current lease and source-derived byte envelope are checked BEFORE decoding
  // a possibly large base64 body. The initial hash cannot grant any authority.
  const capture = { brainId, turnId, sessionId: request.captureSessionId, utteranceId: request.utteranceId,
    sequence: request.sequence, capturedAt: request.capturedAt, bytes, contentSha256: "" };
  requireLease(capture);
  const data = Buffer.from(request.dataBase64, "base64");
  if (data.byteLength !== bytes || data.toString("base64") !== request.dataBase64) throw new Error("Native audio bytes are not canonical base64.");
  const contentSha256 = sha(data), lease = requireLease({ ...capture, contentSha256 });
  const requestId = sha(`${brainId}\0${request.captureSessionId}\0${request.utteranceId}\0${turnId}`);
  const handle = await attachments.stageBytes(brainId, { requestId, data, displayName: "Microphone audio",
    mimeType: request.mimeType, kind: "audio", source: { kind: "captured-audio", captureSessionId: request.captureSessionId,
      utteranceId: request.utteranceId, sequence: request.sequence, capturedAt: request.capturedAt, selectedAt: new Date().toISOString() } }, signal);
  signal?.throwIfAborted();
  if (handle.bytes !== bytes || handle.contentSha256 !== contentSha256 || handle.mimeType !== request.mimeType || handle.kind !== "audio") throw new Error("This audio capture request already owns different immutable bytes.");
  await attachments.bindToTurn(brainId, turnId, [handle.id]);
  const attachment = await attachments.resolveForTurn(brainId, turnId, handle.id);
  requireLease({ ...capture, contentSha256 });
  signal?.throwIfAborted();
  const input = { format: "omni-native-audio-input" as const, version: 1 as const, attachment,
    capture: { sessionId: request.captureSessionId, utteranceId: request.utteranceId, sequence: request.sequence,
      capturedAt: request.capturedAt, permissionRevision: lease.permissionRevision } };
  return verifiedNativeAudioInput({ ...input, inputSha256: sha(nativeAudioIdentityJson(input)) });
}

/** Pending IO cancellation only; this map never stores a microphone grant. */
export class NativeAudioChatPreparations {
  private readonly pending = new Map<string, { brainId: string; turnId: string; controller: AbortController; settled: Promise<void>; settle(): void }>();
  constructor(private readonly attachments: TaskAttachmentService, private readonly requireLease: NativeAudioCaptureLeaseCheck) {}

  async prepare(brainId: string, turnId: string, value: unknown): Promise<NativeAudioInput> {
    const key = `${brainId}\0${turnId}`;
    if (this.pending.has(key)) throw new Error("This audio turn is already being prepared.");
    let settle!: () => void;
    const settled = new Promise<void>(resolve => { settle = resolve; });
    const entry = { brainId, turnId, controller: new AbortController(), settled, settle };
    this.pending.set(key, entry);
    try { return await prepareNativeAudioChat(this.attachments, this.requireLease, brainId, turnId, value, entry.controller.signal); }
    finally { if (this.pending.get(key) === entry) this.pending.delete(key); entry.settle(); }
  }

  cancel(brainId: string, turnId?: string): number {
    let count = 0;
    for (const entry of this.pending.values()) if (entry.brainId === brainId && (turnId === undefined || entry.turnId === turnId) && !entry.controller.signal.aborted) {
      entry.controller.abort(new DOMException("Audio input staging was stopped.", "AbortError")); count += 1;
    }
    return count;
  }
  async cancelAndWait(brainId: string, turnId?: string): Promise<number> {
    const owned = [...this.pending.values()].filter(entry => entry.brainId === brainId && (turnId === undefined || entry.turnId === turnId));
    const count = this.cancel(brainId, turnId);
    await Promise.all(owned.map(entry => entry.settled));
    return count;
  }
}

/** Revalidate current lease AND immutable attachment immediately before dispatch. */
export async function revalidateNativeAudioChat(
  attachments: TaskAttachmentService,
  requireLease: NativeAudioCaptureLeaseCheck,
  brainId: string,
  value: NativeAudioInput
): Promise<void> {
  const input = verifiedNativeAudioInput(value), source = input.attachment, capture = input.capture;
  if (source.brainId !== brainId) throw new Error("Native audio belongs to another mind.");
  const current = await attachments.resolveForTurn(brainId, source.turnId, source.attachmentId);
  if (nativeAudioIdentityJson({ ...input, attachment: current }) !== nativeAudioIdentityJson(input)) throw new Error("Native audio attachment changed while queued.");
  const lease = requireLease({ brainId, turnId: source.turnId, sessionId: capture.sessionId, utteranceId: capture.utteranceId,
    sequence: capture.sequence, capturedAt: capture.capturedAt, bytes: source.bytes, contentSha256: source.contentSha256 });
  if (lease.permissionRevision !== capture.permissionRevision) throw new Error("Native audio microphone authority changed while queued.");
}
