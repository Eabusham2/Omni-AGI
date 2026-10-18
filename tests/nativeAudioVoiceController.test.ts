import { describe, expect, it, vi } from "vitest";
import { LiveVoiceController } from "../src/renderer/src/liveVoiceController";
import type { LiveVoiceNativeAudioHandlers } from "../src/shared/liveVoice";
import type { ChatResult } from "../src/shared/types";
import type { NativeAudioChatRequest } from "../src/shared/nativeAudioChat";

const input: NativeAudioChatRequest = { format: "omni-native-audio-chat", version: 1, captureSessionId: "mic-lease",
  utteranceId: "clip-one", sequence: 0, mimeType: "audio/wav", dataBase64: "UklGRg==", capturedAt: "2026-10-11T14:00:00Z" };
const measured = { listening: true, nativeAudioChatInput: true, voice: false, detail: "Explicit synthetic contract fixture; no model." };

function fixture(buffered = false, inspect: () => Promise<typeof measured> = async () => measured) {
  let handlers: LiveVoiceNativeAudioHandlers | undefined, resolve!: (value: ChatResult) => void;
  const pending = new Promise<ChatResult>(finish => { resolve = finish; });
  const capture = { available: true, request: vi.fn(async () => ({ stop: vi.fn() })) };
  const recognition = { available: false, start: vi.fn() };
  const stop = vi.fn();
  const neural = { inspect: vi.fn(inspect), startListening: vi.fn(async (_id, listener?: LiveVoiceNativeAudioHandlers) => {
    handlers = listener; return { stop };
  }) };
  const chat = { send: vi.fn(), sendAudio: vi.fn(async () => pending), cancel: vi.fn(async () => 1) };
  const accepted = vi.fn();
  const controller = new LiveVoiceController({ brainId: "audio-brain", capture, recognition, neural, chat,
    synthesis: { available: true, speak: vi.fn(), cancel: vi.fn() }, createId: () => "utterance-turn",
    onAcceptedReply: accepted, preferences: { schemaVersion: 1, deliveryMode: buffered ? "buffered" : "live", pace: "normal", neuralListening: true, neuralVoice: false } });
  return { controller, capture, recognition, neural, chat, stop, accepted, get handlers() { return handlers!; }, resolve };
}

describe("native raw audio voice routing (synthetic host, no model)", () => {
  it("starts without any recognizer and sends exact raw audio, never STT text", async () => {
    const f = fixture();
    await f.controller.readyCapabilities(); const reads = f.neural.inspect.mock.calls.length;
    await f.controller.start();
    expect(f.neural.inspect).toHaveBeenCalledTimes(reads);
    expect(f.recognition.start).not.toHaveBeenCalled(); expect(f.capture.request).not.toHaveBeenCalled();
    f.handlers.onSpeechStart(); f.handlers.onFinalAudio(input);
    expect(f.chat.sendAudio).toHaveBeenCalledWith("audio-brain", input, "voice-utterance-turn");
    expect(f.chat.send).not.toHaveBeenCalled();
    expect(f.controller.getState().activeUtterance?.transcript).toBe("");
    expect(JSON.stringify(f.controller.getState().activeUtterance)).not.toContain("dataBase64");
    await f.controller.stop(); expect(f.chat.cancel).toHaveBeenCalledWith("audio-brain", "voice-utterance-turn");
    expect(f.stop).toHaveBeenCalledTimes(1); await f.controller.dispose();
  });
  it("does not substitute platform recognition for assisted-only raw observation", async () => {
    const f = fixture(); f.neural.inspect.mockResolvedValue({ ...measured, nativeAudioChatInput: false });
    await f.controller.start(); expect(f.controller.getState().phase).toBe("unavailable");
    expect(f.neural.startListening).not.toHaveBeenCalled(); expect(f.recognition.start).not.toHaveBeenCalled();
    await f.controller.dispose();
  });
  it("refreshes a constructor probe after explicit Load without opening capture", async () => {
    let loaded = false;
    const f = fixture(false, async () => { if (!loaded) throw new Error("Instance unloaded"); return measured; });
    await Promise.resolve(); loaded = true;
    const status = await f.controller.readyCapabilities();
    expect(status.nativeAudioChatInput).toBe(true);
    expect(f.neural.startListening).not.toHaveBeenCalled(); expect(f.capture.request).not.toHaveBeenCalled();
    await f.controller.dispose();
  });
  it("buffered native output keeps microphone-triggered barge-in disabled", async () => {
    const f = fixture(true); await f.controller.start();
    f.handlers.onSpeechStart(); f.handlers.onFinalAudio(input);
    f.handlers.onSpeechStart(); f.handlers.onFinalAudio({ ...input, utteranceId: "clip-two", sequence: 1 });
    expect(f.chat.sendAudio).toHaveBeenCalledTimes(1); expect(f.chat.cancel).not.toHaveBeenCalled();
    await f.controller.stop(); expect(f.chat.cancel).toHaveBeenCalledTimes(1); await f.controller.dispose();
  });
  it("Stop during capability refresh cannot subsequently start capture", async () => {
    let finish!: (value: typeof measured) => void, first = true;
    const delayed = new Promise<typeof measured>(resolve => { finish = resolve; });
    const f = fixture(false, () => { if (first) { first = false; return delayed; } return Promise.resolve(measured); });
    const ready = f.controller.readyCapabilities(); await Promise.resolve(); await Promise.resolve();
    const start = f.controller.start(); await f.controller.stop();
    finish(measured);
    await ready; await start;
    expect(f.controller.getState().enabled).toBe(false); expect(f.neural.startListening).not.toHaveBeenCalled();
    await f.controller.dispose();
  });
  it("keeps a running route immutable while allowing pace/delivery updates", async () => {
    const f = fixture(); await f.controller.start();
    const preferences = f.controller.getState().preferences;
    expect(() => f.controller.setPreferences({ ...preferences, neuralListening: false })).toThrow(/End the current call/);
    expect(() => f.controller.setPreferences({ ...preferences, pace: "slow", deliveryMode: "buffered" })).not.toThrow();
    await f.controller.stop();
    expect(() => f.controller.setPreferences({ ...preferences, neuralListening: false })).not.toThrow();
    await f.controller.dispose();
  });
});
