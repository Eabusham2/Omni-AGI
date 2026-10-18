import { describe, expect, it, vi } from "vitest";
import { createNativeAudioCaptureAdapter, pcmWave } from "../src/renderer/src/nativeAudioCapture";
import type { LiveVoiceNativeAudioHandlers, LiveVoiceCaptureSession } from "../src/shared/liveVoice";
import type { NativeAudioCaptureAuthorization } from "../src/renderer/src/nativeAudioCapture";

function contextFixture() {
  const processor = { onaudioprocess: null as ((event: AudioProcessingEvent) => void) | null, connect: vi.fn(), disconnect: vi.fn() };
  const source = { connect: vi.fn(), disconnect: vi.fn() }, gain = { gain: { value: 1 }, connect: vi.fn(), disconnect: vi.fn() };
  const close = vi.fn(async () => undefined);
  class Context {
    sampleRate = 16000; destination = {};
    createMediaStreamSource = vi.fn(() => source);
    createScriptProcessor = vi.fn(() => processor);
    createGain = vi.fn(() => gain);
    resume = vi.fn(async () => undefined); close = close;
  }
  return { Context: Context as unknown as typeof AudioContext, processor, close, gain,
    samples: (values: Float32Array) => processor.onaudioprocess?.({ inputBuffer: { getChannelData: () => values } } as unknown as AudioProcessingEvent) };
}
const handlers = (): LiveVoiceNativeAudioHandlers & { onFinalAudio: ReturnType<typeof vi.fn>; onError: ReturnType<typeof vi.fn>; onSpeechStart: ReturnType<typeof vi.fn> } =>
  ({ onFinalAudio: vi.fn(), onError: vi.fn(), onSpeechStart: vi.fn() });

describe("actual PCM capture boundaries with synthetic owned WebAudio handles", () => {
  it("encodes every provided sample into PCM without a recognizer", () => {
    const bytes = pcmWave([new Float32Array([-.5, 0, .5]), new Float32Array([1])], 16000), raw = Buffer.from(bytes);
    expect(raw.subarray(0, 4).toString()).toBe("RIFF"); expect(raw.readUInt32LE(24)).toBe(16000);
    expect(raw.readUInt32LE(40)).toBe(8); expect(raw.length).toBe(52);
    expect(raw.readInt16LE(44)).toBe(-16384); expect(raw.readInt16LE(50)).toBe(32767);
    expect(() => pcmWave([new Float32Array([Number.NaN])], 16000)).toThrow(/nonfinite/);
  });
  it("segments real supplied PCM and closes exactly the owned stream/lease", async () => {
    const context = contextFixture(), stopMic = vi.fn(), stopLease = vi.fn(async () => undefined), observer = handlers();
    const capture = { available: true, request: vi.fn(async () => ({ stream: {} as MediaStream, stop: stopMic })) };
    const adapter = createNativeAudioCaptureAdapter(capture, context.Context);
    const authorize = vi.fn(async () => ({ sessionId: "actual-owned-lease", maxBytes: 100000, stop: stopLease }));
    const session = await adapter.start(authorize, observer);
    context.samples(new Float32Array(1600).fill(.25)); context.samples(new Float32Array(11200));
    expect(observer.onSpeechStart).toHaveBeenCalledTimes(1); expect(observer.onFinalAudio).toHaveBeenCalledTimes(1);
    const actual = observer.onFinalAudio.mock.calls[0]![0];
    expect(actual.captureSessionId).toBe("actual-owned-lease"); expect(actual.mimeType).toBe("audio/wav");
    const raw = Buffer.from(actual.dataBase64, "base64"); expect(raw.readUInt32LE(40)).toBe(12800 * 2);
    expect(actual).not.toHaveProperty("transcript"); expect(context.gain.gain.value).toBe(0);
    session.stop(); session.stop();
    expect(stopMic).toHaveBeenCalledTimes(1); expect(stopLease).toHaveBeenCalledTimes(1); expect(context.close).toHaveBeenCalledTimes(1);
  });
  it("resource envelope overflow pauses instead of sending a truncated utterance", async () => {
    const context = contextFixture(), observer = handlers(), stop = vi.fn();
    const adapter = createNativeAudioCaptureAdapter({ available: true, request: vi.fn(async () => ({ stream: {} as MediaStream, stop })) }, context.Context);
    await adapter.start(async () => ({ sessionId: "owned-lease", maxBytes: 100, stop: vi.fn(async () => undefined) }), observer);
    context.samples(new Float32Array(64).fill(.25));
    expect(observer.onError).toHaveBeenCalledTimes(1); expect(observer.onFinalAudio).not.toHaveBeenCalled(); expect(stop).toHaveBeenCalledTimes(1);
  });
  it("Stop closes a late OS microphone result without creating a main lease", async () => {
    const context = contextFixture(), observer = handlers(), stop = vi.fn(), controller = new AbortController();
    let resolve!: (value: LiveVoiceCaptureSession) => void;
    const capture = { available: true, request: vi.fn(() => new Promise<LiveVoiceCaptureSession>(finish => { resolve = finish; })) };
    const authorize = vi.fn(async () => ({ sessionId: "late-lease", maxBytes: 100000, stop: vi.fn(async () => undefined) }));
    const pending = createNativeAudioCaptureAdapter(capture, context.Context).start(authorize, observer, controller.signal);
    controller.abort(); await expect(pending).rejects.toBeDefined();
    resolve({ stream: {} as MediaStream, stop }); await Promise.resolve();
    expect(stop).toHaveBeenCalledTimes(1); expect(authorize).not.toHaveBeenCalled();
  });
  it("Stop releases microphone immediately while a late main authorization is still queued", async () => {
    const context = contextFixture(), observer = handlers(), stop = vi.fn(), controller = new AbortController(), stopLease = vi.fn(async () => undefined);
    let resolve!: (value: NativeAudioCaptureAuthorization) => void;
    const authorization = new Promise<NativeAudioCaptureAuthorization>(finish => { resolve = finish; });
    const authorize = vi.fn(async () => authorization);
    const pending = createNativeAudioCaptureAdapter({ available: true, request: vi.fn(async () => ({ stream: {} as MediaStream, stop })) }, context.Context).start(authorize, observer, controller.signal);
    await vi.waitFor(() => expect(authorize).toHaveBeenCalledTimes(1));
    controller.abort(); await expect(pending).rejects.toBeDefined();
    expect(stop).toHaveBeenCalledTimes(1);
    resolve({ sessionId: "late-main-lease", maxBytes: 100000, stop: stopLease }); await Promise.resolve(); await Promise.resolve();
    expect(stopLease).toHaveBeenCalledTimes(1); expect(observer.onFinalAudio).not.toHaveBeenCalled();
  });
  it("an already stopped start cannot open an OS permission request", async () => {
    const context = contextFixture(), controller = new AbortController(); controller.abort();
    const capture = { available: true, request: vi.fn(async () => ({ stream: {} as MediaStream, stop: vi.fn() })) };
    const authorize = vi.fn(async () => ({ sessionId: "owned-lease", maxBytes: 1000, stop: vi.fn(async () => undefined) }));
    await expect(createNativeAudioCaptureAdapter(capture, context.Context).start(authorize, handlers(), controller.signal)).rejects.toBeDefined();
    expect(capture.request).not.toHaveBeenCalled(); expect(authorize).not.toHaveBeenCalled();
  });
});
