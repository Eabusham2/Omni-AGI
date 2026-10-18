import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { VoiceRoutingControls } from "../src/renderer/src/VoiceRoutingControls";
import { savedLegacyVoice } from "../src/renderer/src/useInstanceVoice";
import { DEFAULT_INSTANCE_VOICE_SETTINGS, type InstanceVoiceSnapshot } from "../src/shared/voiceRouting";

const snapshot: InstanceVoiceSnapshot = { brainId: "voice-ui", revision: 1, updatedAt: null,
  preferences: { ...DEFAULT_INSTANCE_VOICE_SETTINGS }, initialized: true, legacyMigrationAllowed: false,
  inheritedVariants: 0, parameterEvidence: null, parameterDetail: "No verified count yet." };

describe("instance voice controls without capture or models", () => {
  it("shows independent automatic/manual input and output before starting a call", () => {
    const html = renderToStaticMarkup(<VoiceRoutingControls snapshot={snapshot} callActive={false}
      utteranceActive={false} onSave={async () => undefined} />);
    expect(html).toContain("This brain’s raw audio input"); expect(html).toContain("This brain’s audio output");
    expect(html).toContain("Automatic · platform recognition"); expect(html).toContain("Automatic · by measured size");
    expect(html).toContain("platform speech through 200B"); expect(html).toContain("No verified count yet.");
    expect(html).not.toContain("neuralListening"); expect(html).not.toContain("getUserMedia");
  });
  it("locks route changes during a call and delivery during an active utterance", () => {
    const html = renderToStaticMarkup(<VoiceRoutingControls snapshot={snapshot} callActive utteranceActive onSave={async () => undefined} />);
    expect(html.match(/<select[^>]*disabled/g)).toHaveLength(4);
    expect(html).toContain("End the current call"); expect(html).toContain("Wait for the full reply");
    expect(html).not.toContain("checkbox");
  });
  it("does not convert absent or corrupt storage into explicit manual settings", () => {
    for (const raw of [null, "", "{", "{}", JSON.stringify({ schemaVersion: 1, neuralVoice: true })]) {
      expect(savedLegacyVoice(raw)).toBeUndefined();
    }
  });
  it("preserves a real old record including false choices only when the host allows migration", () => {
    const value = { schemaVersion: 1, deliveryMode: "buffered", pace: "fast", neuralListening: false, neuralVoice: true };
    expect(savedLegacyVoice(JSON.stringify(value))).toEqual(value);
    expect(savedLegacyVoice(JSON.stringify({ ...value, pace: "invented" }))).toBeUndefined();
  });
});
