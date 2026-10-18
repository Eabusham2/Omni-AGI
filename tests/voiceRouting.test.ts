import { describe, expect, it } from "vitest";
import { DEFAULT_INSTANCE_VOICE_SETTINGS, NATIVE_SPEECH_OUTPUT_THRESHOLD, instanceVoiceSettingsFromLegacy,
  resolveInstanceVoice, validateInstanceVoiceSettings, type NativeVoiceParameterEvidence } from "../src/shared/voiceRouting";
import type { LiveVoiceCapabilities } from "../src/shared/liveVoice";

const evidence = (count: number): NativeVoiceParameterEvidence => ({ source: "committed-native-parameter-accounting",
  totalNeuralParameters: count, countingRule: "distinct-native-logical-parameters",
  metadataSha256: "a".repeat(64), substrateGenerationSha256: "b".repeat(64), mutableGenerationSha256: "c".repeat(64),
  parameterChecksum: "d".repeat(64), measuredAt: "2026-10-11T10:00:00Z" });
const caps: LiveVoiceCapabilities & { nativeAudioChatInput: boolean } = { microphoneCapture: true, speechRecognition: true,
  speechSynthesis: true, fullDuplex: true, bargeIn: true, neuralListening: true, neuralVoice: true,
  nativeAudioChatInput: true };

describe("instance-owned native speech output selection", () => {
  it.each([0, NATIVE_SPEECH_OUTPUT_THRESHOLD - 1, NATIVE_SPEECH_OUTPUT_THRESHOLD])("keeps platform output through200B (%s)", count => {
    const selected = resolveInstanceVoice({ ...DEFAULT_INSTANCE_VOICE_SETTINGS }, evidence(count), caps);
    expect(selected.output.route).toBe("platform"); expect(selected.output.reason).toBe("automatic-through-200b");
    expect(selected.input.route).toBe("platform"); expect(selected.ready).toBe(true);
  });
  it("selects native output strictly above200B while automatic microphone input remains platform", () => {
    const selected = resolveInstanceVoice({ ...DEFAULT_INSTANCE_VOICE_SETTINGS }, evidence(NATIVE_SPEECH_OUTPUT_THRESHOLD + 1), caps);
    expect(selected.output.route).toBe("native"); expect(selected.input.route).toBe("platform");
    expect(selected.controllerPreferences?.neuralVoice).toBe(true); expect(selected.controllerPreferences?.neuralListening).toBe(false);
  });
  it("does not replace either independent manual choice when size or the other choice changes", () => {
    const selected = resolveInstanceVoice({ ...DEFAULT_INSTANCE_VOICE_SETTINGS, input: "native", output: "platform", pace: "slow", deliveryMode: "buffered" },
      evidence(NATIVE_SPEECH_OUTPUT_THRESHOLD * 2), caps);
    expect(selected.input.route).toBe("native"); expect(selected.output.route).toBe("platform");
    expect(selected.controllerPreferences).toEqual({ schemaVersion: 1, neuralListening: true, neuralVoice: false, pace: "slow", deliveryMode: "buffered" });
    expect(resolveInstanceVoice({ ...DEFAULT_INSTANCE_VOICE_SETTINGS, output: "native" }, evidence(1), caps).output.route).toBe("native");
  });
  it("keeps unavailable native output selected and never falls back to available platform TTS", () => {
    const selected = resolveInstanceVoice({ ...DEFAULT_INSTANCE_VOICE_SETTINGS }, evidence(NATIVE_SPEECH_OUTPUT_THRESHOLD + 1), { ...caps, neuralVoice: false });
    expect(selected.output.route).toBe("native"); expect(selected.output.available).toBe(false); expect(selected.ready).toBe(false);
    expect(selected.output.detail).toContain("not substituted"); expect(selected.controllerPreferences?.neuralVoice).toBe(true);
  });
  it("does not call assisted audio observation plus STT a direct native conversation route", () => {
    const selected = resolveInstanceVoice({ ...DEFAULT_INSTANCE_VOICE_SETTINGS, input: "native" }, evidence(1), { ...caps, nativeAudioChatInput: false });
    expect(selected.input.available).toBe(false); expect(selected.input.route).toBe("native"); expect(selected.input.detail).toContain("not a native conversation route");
    // A genuinely supported native input does not require a platform transcript.
    expect(resolveInstanceVoice({ ...DEFAULT_INSTANCE_VOICE_SETTINGS, input: "native" }, evidence(1), { ...caps, speechRecognition: false }).input.available).toBe(true);
  });
  it("leaves unknown/unverified Auto output unresolved without assuming a platform fallback", () => {
    for (const value of [null, { ...evidence(10), metadataSha256: "unverified" }, evidence(Number.NaN)]) {
      const selected = resolveInstanceVoice({ ...DEFAULT_INSTANCE_VOICE_SETTINGS }, value, caps);
      expect(selected.output.route).toBeNull(); expect(selected.controllerPreferences).toBeNull(); expect(selected.ready).toBe(false);
    }
    expect(resolveInstanceVoice({ ...DEFAULT_INSTANCE_VOICE_SETTINGS, output: "platform" }, null, caps).output.available).toBe(true);
  });
  it("preserves stored legacy booleans as explicit manual selections and rejects behavioral fields", () => {
    expect(instanceVoiceSettingsFromLegacy({ schemaVersion: 1, deliveryMode: "buffered", pace: "fast", neuralListening: true, neuralVoice: false }))
      .toEqual({ schemaVersion: 1, deliveryMode: "buffered", pace: "fast", input: "native", output: "platform" });
    expect(() => validateInstanceVoiceSettings({ ...DEFAULT_INSTANCE_VOICE_SETTINGS, prompt: "override" })).toThrow(/Invalid/);
  });
});
