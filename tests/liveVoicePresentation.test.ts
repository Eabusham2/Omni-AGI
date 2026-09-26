import { describe, expect, it } from "vitest";
import type { LiveVoiceState } from "../src/shared/liveVoice";
import { liveVoiceStatusCopy } from "../src/renderer/src/liveVoicePresentation";

const base = {
  schemaVersion: 1,
  enabled: true,
  phase: "listening",
  capabilities: {
    microphoneCapture: true,
    speechRecognition: true,
    speechSynthesis: true,
    fullDuplex: true,
    bargeIn: true,
    neuralListening: false,
    neuralVoice: false
  },
  preferences: {
    schemaVersion: 1,
    deliveryMode: "live",
    pace: "normal",
    neuralListening: false,
    neuralVoice: false
  },
  capture: "active",
  recognition: "listening",
  synthesis: "idle",
  interimTranscript: "",
  speechBackpressure: {
    queuedChunks: 0,
    pendingCharacters: 0,
    skippedCharacters: 0,
    maxQueuedChunks: 3
  },
  hiddenBehavioralPrompt: false
} as const satisfies LiveVoiceState;

describe("live voice presentation", () => {
  it("uses Ponder as the single recurrent-computation label", () => {
    expect(liveVoiceStatusCopy({ ...base, phase: "pondering" })).toMatchObject({
      title: "Pondering the utterance"
    });
  });

  it("makes duplex barge-in explicit while speaking", () => {
    expect(liveVoiceStatusCopy({ ...base, phase: "speaking", synthesis: "speaking" })).toEqual({
      title: "Speaking the brain reply",
      detail: "The microphone remains live; speaking now immediately barges in.",
      tone: "active"
    });
  });

  it("does not promise voice-triggered barge-in for buffered delivery", () => {
    const copy = liveVoiceStatusCopy({
      ...base,
      phase: "speaking",
      synthesis: "speaking",
      preferences: { ...base.preferences, deliveryMode: "buffered" }
    });
    expect(copy.detail).toBe(
      "Buffered delivery is speaking the complete reply. Use Stop to cancel it."
    );
    expect(copy.detail).not.toContain("speaking now immediately barges in");
  });

  it("shows the real unsupported reason instead of implying transcription", () => {
    expect(
      liveVoiceStatusCopy({
        ...base,
        enabled: false,
        phase: "unavailable",
        capture: "idle",
        recognition: "unavailable",
        error: {
          code: "recognition-unavailable",
          message: "No platform recognizer; no transcript is fabricated.",
          recoverable: false
        }
      })
    ).toEqual({
      title: "Live voice unavailable",
      detail: "No platform recognizer; no transcript is fabricated.",
      tone: "warning"
    });
  });
});
