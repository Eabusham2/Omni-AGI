import { createHash } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import { checkedNativeAudioInfo, nativeAudioIdentityJson, nativeAudioMessageInfo, type NativeAudioChatRequest,
  type NativeAudioInputReceipt } from "../src/shared/nativeAudioChat";
import { NativeAudioChatPreparations, prepareNativeAudioChat, revalidateNativeAudioChat } from "../src/main/nativeAudioChat";
import { expectedChatInputHash, observedChatInputHash, verifiedNativeAudioInput } from "../src/main/nativeAudioIdentity";
import { receivedChatInput, receivedChatInputFromLedger } from "../src/main/chatInputReceipt";
import type { TaskAttachmentService } from "../src/main/taskAttachmentService";
import { pcmWave } from "../src/renderer/src/nativeAudioCapture";

const sha = (value: string | Uint8Array) => createHash("sha256").update(value).digest("hex");
const payload = pcmWave([new Float32Array([0, .25, -.25, .5])], 16000);
const request = (): NativeAudioChatRequest => ({ format: "omni-native-audio-chat", version: 1, captureSessionId: "mic-lease",
  utteranceId: "clip-one", sequence: 0, mimeType: "audio/wav", dataBase64: Buffer.from(payload).toString("base64"), capturedAt: "2026-10-11T14:00:00Z" });

function staging() {
  let active = true, storedHash: string | undefined;
  const attachment = { brainId: "audio-brain", turnId: "audio-turn", attachmentId: "owned-audio", contentSha256: sha(payload),
    bytes: payload.length, mimeType: "audio/wav", kind: "audio", relativePath: `attachments/blobs/${sha(payload)}.data`,
    displayName: "Microphone audio", manifestSha256: "b".repeat(64) };
  const stageBytes = vi.fn(async (_brainId, value, _signal?: AbortSignal) => {
    storedHash ??= sha(value.data);
    return { id: attachment.attachmentId, ...attachment, contentSha256: storedHash };
  });
  const bindToTurn = vi.fn(async () => [attachment]), resolveForTurn = vi.fn(async () => ({ ...attachment }));
  const attachments = { stageBytes, bindToTurn, resolveForTurn } as unknown as TaskAttachmentService;
  const requireLease = vi.fn((value) => {
    if (!active || value.sessionId !== "mic-lease" || value.brainId !== "audio-brain" || value.bytes > 1000) throw new Error("No current microphone lease");
    return { permissionRevision: 1, maxBytes: 1000 };
  });
  return { attachments, requireLease, stageBytes, bindToTurn, resolveForTurn, revoke: () => { active = false; } };
}

function receipt(input: Awaited<ReturnType<typeof prepareNativeAudioChat>>): NativeAudioInputReceipt {
  return { input, assemblyId: "audio-assembly", sourceBytes: payload.length, sourceSha256: sha(payload), decodedWindows: 1,
    modelSamples: 4, modelSampleRate: 16000, wholeInputTraversed: true, nativeSensory: true, transcriptUsed: false };
}

describe("same-brain audio input protocol (no model)", () => {
  it("stages raw bytes through one owned registry and binds an audio identity, never a display caption", async () => {
    const fixture = staging(), input = await prepareNativeAudioChat(fixture.attachments, fixture.requireLease, "audio-brain", "audio-turn", request());
    expect(fixture.stageBytes).toHaveBeenCalledTimes(1); expect(fixture.bindToTurn).toHaveBeenCalledWith("audio-brain", "audio-turn", ["owned-audio"]);
    expect(input.inputSha256).toBe(sha(nativeAudioIdentityJson(input)));
    expect(input.inputSha256).not.toBe(sha("")); expect(input.inputSha256).not.toBe(sha("Microphone audio"));
    expect(expectedChatInputHash("", input)).toBe(input.inputSha256);
    expect(() => expectedChatInputHash("fabricated transcript", input)).toThrow(/transcript/);
  });
  it("refuses missing/expired current microphone lease before source decode or stage", async () => {
    const fixture = staging(); fixture.revoke();
    await expect(prepareNativeAudioChat(fixture.attachments, fixture.requireLease, "audio-brain", "audio-turn", request())).rejects.toThrow(/lease/);
    expect(fixture.stageBytes).not.toHaveBeenCalled();
  });
  it("rechecks authority and exact immutable bytes after queue wait", async () => {
    const fixture = staging(), input = await prepareNativeAudioChat(fixture.attachments, fixture.requireLease, "audio-brain", "audio-turn", request());
    await revalidateNativeAudioChat(fixture.attachments, fixture.requireLease, "audio-brain", input);
    fixture.revoke();
    await expect(revalidateNativeAudioChat(fixture.attachments, fixture.requireLease, "audio-brain", input)).rejects.toThrow(/lease/);
    fixture.resolveForTurn.mockResolvedValueOnce({ ...input.attachment, manifestSha256: "c".repeat(64) });
    await expect(revalidateNativeAudioChat(fixture.attachments, fixture.requireLease, "audio-brain", input)).rejects.toThrow(/changed/);
  });
  it("same captured retry has exact identity and different bytes cannot replace an existing stage", async () => {
    const fixture = staging(), first = await prepareNativeAudioChat(fixture.attachments, fixture.requireLease, "audio-brain", "audio-turn", request());
    expect(await prepareNativeAudioChat(fixture.attachments, fixture.requireLease, "audio-brain", "audio-turn", request())).toEqual(first);
    const changed = Buffer.from(payload); changed[45] = changed[45]! ^ 1;
    await expect(prepareNativeAudioChat(fixture.attachments, fixture.requireLease, "audio-brain", "audio-turn", { ...request(), dataBase64: changed.toString("base64") })).rejects.toThrow(/different immutable bytes/);
  });
  it("rejects path, source hash, transcript and raw fields in the public projection", async () => {
    const fixture = staging(), input = await prepareNativeAudioChat(fixture.attachments, fixture.requireLease, "audio-brain", "audio-turn", request());
    expect(() => verifiedNativeAudioInput({ ...input, attachment: { ...input.attachment, relativePath: "../secret" } })).toThrow();
    expect(() => verifiedNativeAudioInput({ ...input, capture: { ...input.capture, sequence: 1 } })).toThrow(/checksum/);
    const info = nativeAudioMessageInfo(receipt(input));
    expect(checkedNativeAudioInfo(info, input)).toEqual(info);
    expect(JSON.stringify(info)).not.toMatch(/relativePath|permissionRevision|dataBase64/);
    expect(() => checkedNativeAudioInfo({ ...info, relativePath: input.attachment.relativePath }, input)).toThrow();
  });
  it("normalizes genuine accepted native receipt and revalidates its safe streamed projection", async () => {
    const fixture = staging(), input = await prepareNativeAudioChat(fixture.attachments, fixture.requireLease, "audio-brain", "audio-turn", request());
    const raw = { committed: true, turnId: "audio-turn", inputSha256: input.inputSha256, humanMessageId: "human-one", afterimageId: "afterimage-one",
      createdAt: "2026-10-11T14:00:01Z", attentionEpoch: 0, nativeAudioInput: receipt(input) };
    const expected = { turnId: "audio-turn", input: "", nativeAudioInput: input };
    const result = receivedChatInput(raw, expected);
    expect(result.humanMessage.content).toBe(""); expect(result.humanMessage.nativeAudioInput?.transcriptUsed).toBe(false);
    expect(JSON.stringify(result)).not.toMatch(/relativePath|permissionRevision|dataBase64/);
    expect(receivedChatInput(result.inputReceipt, expected)).toEqual(result);
    expect(() => receivedChatInput(raw, { turnId: "audio-turn", input: "" })).toThrow();
    expect(observedChatInputHash("", raw.nativeAudioInput)).toBe(input.inputSha256);
  });
  it("ledger retry preserves original audio source identity without mic re-admission or text hash", async () => {
    const fixture = staging(), input = await prepareNativeAudioChat(fixture.attachments, fixture.requireLease, "audio-brain", "audio-turn", request());
    const raw = { committed: true, turnId: "audio-turn", inputSha256: input.inputSha256, humanMessageId: "human-one", afterimageId: "afterimage-one",
      createdAt: "2026-10-11T14:00:01Z", attentionEpoch: 0, nativeAudioInput: receipt(input) };
    const human = { id: "human-one", role: "human", content: "", turn_id: "audio-turn", created_at: raw.createdAt,
      attention_epoch: 0, input_accepted_before_reply: true, native_audio_input: raw.nativeAudioInput };
    const result = receivedChatInputFromLedger(raw, human);
    expect(result.inputReceipt.inputSha256).toBe(input.inputSha256);
    expect(result.humanMessage.nativeAudioInput?.sourceBrainId).toBe("audio-brain");
    expect(() => receivedChatInputFromLedger(raw, { ...human, content: "Audio message" })).toThrow(/transcript/);
    expect(() => receivedChatInputFromLedger(raw, { ...human, native_audio_input: { ...raw.nativeAudioInput, transcriptUsed: true } })).toThrow();
  });
  it("Stop aborts only this owned pending audio stage and waits for its IO boundary", async () => {
    const fixture = staging();
    fixture.stageBytes.mockImplementationOnce(async (_brainId, _request, signal) => new Promise((_resolve, reject) => {
      signal!.addEventListener("abort", () => reject(signal!.reason), { once: true });
    }));
    const preparations = new NativeAudioChatPreparations(fixture.attachments, fixture.requireLease);
    const pending = preparations.prepare("audio-brain", "audio-turn", request());
    const rejection = pending.catch(error => error);
    await vi.waitFor(() => expect(fixture.stageBytes).toHaveBeenCalledTimes(1));
    expect(await preparations.cancelAndWait("other-brain", "audio-turn")).toBe(0);
    expect(await preparations.cancelAndWait("audio-brain", "audio-turn")).toBe(1);
    expect(await rejection).toMatchObject({ name: "AbortError" }); expect(fixture.bindToTurn).not.toHaveBeenCalled();
  });
});
