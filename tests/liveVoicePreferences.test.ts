import { describe, expect, it } from "vitest";
import {
  DEFAULT_LIVE_VOICE_PREFERENCES,
  LIVE_VOICE_STORAGE_KEY,
  loadLiveVoicePreferences,
  parseLiveVoicePreferences,
  saveLiveVoicePreferences,
  type LiveVoiceStorage
} from "../src/renderer/src/liveVoicePreferences";

class MemoryStorage implements LiveVoiceStorage {
  values = new Map<string, string>();
  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }
  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }
}

describe("live voice presentation preferences", () => {
  it("defaults safely for absent, corrupt, or unknown schemas", () => {
    const storage = new MemoryStorage();
    expect(loadLiveVoicePreferences(storage)).toEqual(DEFAULT_LIVE_VOICE_PREFERENCES);
    storage.values.set(LIVE_VOICE_STORAGE_KEY, "not json");
    expect(loadLiveVoicePreferences(storage)).toEqual(DEFAULT_LIVE_VOICE_PREFERENCES);
    expect(
      parseLiveVoicePreferences({ schemaVersion: 99, deliveryMode: "buffered", pace: "slow" })
    ).toEqual(DEFAULT_LIVE_VOICE_PREFERENCES);
  });

  it("persists typed delivery and pace without brain or behavior fields", () => {
    const storage = new MemoryStorage();
    saveLiveVoicePreferences(storage, {
      schemaVersion: 1,
      deliveryMode: "buffered",
      pace: "fast",
      neuralListening: false,
      neuralVoice: true
    });
    expect(loadLiveVoicePreferences(storage)).toEqual({
      schemaVersion: 1,
      deliveryMode: "buffered",
      pace: "fast",
      neuralListening: false,
      neuralVoice: true
    });
    expect(storage.values.get(LIVE_VOICE_STORAGE_KEY)).not.toMatch(
      /prompt|behavior|personality|brainId/
    );
  });

  it("repairs invalid individual choices", () => {
    expect(
      parseLiveVoicePreferences({
        schemaVersion: 1,
        deliveryMode: "instant",
        pace: "maximum"
      })
    ).toEqual(DEFAULT_LIVE_VOICE_PREFERENCES);
  });
});
