import {
  LIVE_VOICE_PREFERENCES_SCHEMA_VERSION,
  type LiveVoiceDeliveryMode,
  type LiveVoicePace,
  type LiveVoicePreferences
} from "../../shared/liveVoice";

export const LIVE_VOICE_STORAGE_KEY = "omni.liveVoice.v1";

export const DEFAULT_LIVE_VOICE_PREFERENCES: LiveVoicePreferences = {
  schemaVersion: LIVE_VOICE_PREFERENCES_SCHEMA_VERSION,
  deliveryMode: "live",
  pace: "normal",
  neuralListening: false,
  neuralVoice: false
};

export interface LiveVoiceStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

const deliveryModes = new Set<LiveVoiceDeliveryMode>(["live", "buffered"]);
const paces = new Set<LiveVoicePace>(["slow", "normal", "fast"]);

export function parseLiveVoicePreferences(value: unknown): LiveVoicePreferences {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return { ...DEFAULT_LIVE_VOICE_PREFERENCES };
  }
  const candidate = value as Partial<LiveVoicePreferences>;
  if (candidate.schemaVersion !== LIVE_VOICE_PREFERENCES_SCHEMA_VERSION) {
    return { ...DEFAULT_LIVE_VOICE_PREFERENCES };
  }
  return {
    schemaVersion: LIVE_VOICE_PREFERENCES_SCHEMA_VERSION,
    deliveryMode: deliveryModes.has(candidate.deliveryMode as LiveVoiceDeliveryMode)
      ? (candidate.deliveryMode as LiveVoiceDeliveryMode)
      : DEFAULT_LIVE_VOICE_PREFERENCES.deliveryMode,
    pace: paces.has(candidate.pace as LiveVoicePace)
      ? (candidate.pace as LiveVoicePace)
      : DEFAULT_LIVE_VOICE_PREFERENCES.pace,
    neuralListening: candidate.neuralListening === true,
    neuralVoice: candidate.neuralVoice === true
  };
}

export function loadLiveVoicePreferences(
  storage: LiveVoiceStorage
): LiveVoicePreferences {
  const stored = storage.getItem(LIVE_VOICE_STORAGE_KEY);
  if (!stored) return { ...DEFAULT_LIVE_VOICE_PREFERENCES };
  try {
    return parseLiveVoicePreferences(JSON.parse(stored));
  } catch {
    return { ...DEFAULT_LIVE_VOICE_PREFERENCES };
  }
}

export function saveLiveVoicePreferences(
  storage: LiveVoiceStorage,
  preferences: LiveVoicePreferences
): LiveVoicePreferences {
  const normalized = parseLiveVoicePreferences(preferences);
  storage.setItem(LIVE_VOICE_STORAGE_KEY, JSON.stringify(normalized));
  return normalized;
}

export const LIVE_VOICE_PACES: ReadonlyArray<LiveVoicePace> = [
  "slow",
  "normal",
  "fast"
];

export const LIVE_VOICE_DELIVERY_MODES: ReadonlyArray<LiveVoiceDeliveryMode> = [
  "live",
  "buffered"
];
