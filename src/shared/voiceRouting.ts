import type { LiveVoiceCapabilities, LiveVoicePreferences, LiveVoiceDeliveryMode, LiveVoicePace } from "./liveVoice";

export const NATIVE_SPEECH_OUTPUT_THRESHOLD = 200_000_000_000;
export type VoiceRoutePreference = "auto" | "platform" | "native";

/** Host presentation settings, never behavioral instructions or learned state. */
export interface InstanceVoiceSettings {
  schemaVersion: 1;
  input: VoiceRoutePreference;
  output: VoiceRoutePreference;
  deliveryMode: LiveVoiceDeliveryMode;
  pace: LiveVoicePace;
}
export const DEFAULT_INSTANCE_VOICE_SETTINGS: Readonly<InstanceVoiceSettings> = Object.freeze({
  schemaVersion: 1, input: "auto", output: "auto", deliveryMode: "live", pace: "normal"
});

export interface InstanceVoicePreferences {
  brainId: string;
  revision: number;
  updatedAt: string | null;
  preferences: InstanceVoiceSettings;
  initialized: boolean;
  /** Only an existing local instance with no instance-owned record is eligible. */
  legacyMigrationAllowed: boolean;
  inheritedVariants: number;
}

export interface NativeVoiceParameterEvidence {
  source: "committed-native-parameter-accounting";
  totalNeuralParameters: number;
  countingRule: string;
  metadataSha256: string;
  substrateGenerationSha256: string;
  mutableGenerationSha256: string;
  parameterChecksum: string;
  measuredAt: string;
}
export interface InstanceVoiceSnapshot extends InstanceVoicePreferences {
  parameterEvidence: NativeVoiceParameterEvidence | null;
  parameterDetail: string;
}
export interface InstanceVoiceUpdateRequest {
  brainId: string;
  expectedRevision: number;
  preferences: InstanceVoiceSettings;
}
export interface InstanceVoiceLegacyRequest {
  brainId: string;
  legacy: LiveVoicePreferences;
}
export interface InstanceVoiceApi {
  preferences(brainId: string): Promise<InstanceVoiceSnapshot>;
  update(request: InstanceVoiceUpdateRequest): Promise<InstanceVoiceSnapshot>;
  migrateLegacy(request: InstanceVoiceLegacyRequest): Promise<InstanceVoiceSnapshot>;
}

const keys = ["schemaVersion", "input", "output", "deliveryMode", "pace"];
export function validateInstanceVoiceSettings(value: unknown): InstanceVoiceSettings {
  const candidate = value as Partial<InstanceVoiceSettings> | null;
  if (!candidate || typeof candidate !== "object" || Array.isArray(candidate) ||
      Object.keys(candidate).some(key => !keys.includes(key)) || Object.keys(candidate).length !== keys.length ||
      candidate.schemaVersion !== 1 || !["auto", "platform", "native"].includes(String(candidate.input)) ||
      !["auto", "platform", "native"].includes(String(candidate.output)) ||
      !["live", "buffered"].includes(String(candidate.deliveryMode)) ||
      !["slow", "normal", "fast"].includes(String(candidate.pace))) {
    throw new Error("Invalid instance voice preferences.");
  }
  return { schemaVersion: 1, input: candidate.input!, output: candidate.output!,
    deliveryMode: candidate.deliveryMode!, pace: candidate.pace! };
}

/** A stored legacy false is an explicit platform choice, not a new Auto default. */
export function instanceVoiceSettingsFromLegacy(value: unknown): InstanceVoiceSettings {
  const candidate = value as Partial<LiveVoicePreferences> | null;
  if (!candidate || candidate.schemaVersion !== 1 ||
      typeof candidate.neuralListening !== "boolean" || typeof candidate.neuralVoice !== "boolean") {
    throw new Error("Invalid saved legacy voice preferences.");
  }
  return validateInstanceVoiceSettings({ schemaVersion: 1,
    input: candidate.neuralListening ? "native" : "platform",
    output: candidate.neuralVoice ? "native" : "platform",
    deliveryMode: candidate.deliveryMode, pace: candidate.pace });
}

export interface ResolvedVoiceRoute {
  requested: VoiceRoutePreference;
  route: "platform" | "native" | null;
  reason: "manual" | "platform-auto-input" | "automatic-through-200b" |
    "automatic-above-200b" | "awaiting-native-accounting";
  available: boolean;
  detail: string;
}
export interface ResolvedInstanceVoice {
  input: ResolvedVoiceRoute;
  output: ResolvedVoiceRoute;
  ready: boolean;
  /** Null means no output route was determined; do not start a voice session. */
  controllerPreferences: LiveVoicePreferences | null;
}

/** The size threshold applies to output only. No unavailable route falls back. */
export function resolveInstanceVoice(
  settings: InstanceVoiceSettings,
  evidence: NativeVoiceParameterEvidence | null,
  capabilities: LiveVoiceCapabilities & { nativeAudioChatInput?: boolean }
): ResolvedInstanceVoice {
  const selected = validateInstanceVoiceSettings(settings);
  const completeEvidence = evidence?.source === "committed-native-parameter-accounting" &&
    [evidence.metadataSha256, evidence.substrateGenerationSha256, evidence.mutableGenerationSha256,
      evidence.parameterChecksum].every(value => typeof value === "string" && /^[a-f0-9]{64}$/.test(value)) &&
    typeof evidence.countingRule === "string" && !!evidence.countingRule.trim() &&
    typeof evidence.measuredAt === "string" && Number.isFinite(Date.parse(evidence.measuredAt));
  const count = completeEvidence && evidence &&
    Number.isSafeInteger(evidence.totalNeuralParameters) && evidence.totalNeuralParameters >= 0
    ? evidence.totalNeuralParameters : null;
  const inputRoute = selected.input === "native" ? "native" : "platform";
  const outputRoute = selected.output === "auto"
    ? count === null ? null : count > NATIVE_SPEECH_OUTPUT_THRESHOLD ? "native" : "platform"
    : selected.output;
  const inputAvailable = capabilities.microphoneCapture && (inputRoute === "platform"
    ? capabilities.speechRecognition : capabilities.neuralListening && capabilities.nativeAudioChatInput === true);
  const outputAvailable = outputRoute === "native" ? capabilities.neuralVoice
    : outputRoute === "platform" ? capabilities.speechSynthesis : false;
  const input: ResolvedVoiceRoute = { requested: selected.input, route: inputRoute,
    reason: selected.input === "auto" ? "platform-auto-input" : "manual", available: inputAvailable,
    detail: inputAvailable ? inputRoute === "native"
      ? "Same-brain raw audio conversation input is selected; no platform transcript is substituted."
      : "Platform speech recognition is selected."
      : inputRoute === "native" ? "Direct same-brain audio conversation input is unavailable. An assisted raw-audio observation plus platform transcript is not a native conversation route; no platform fallback was substituted."
        : "Platform microphone capture or speech recognition is unavailable." };
  const output: ResolvedVoiceRoute = { requested: selected.output, route: outputRoute,
    reason: selected.output !== "auto" ? "manual" : count === null ? "awaiting-native-accounting"
      : count > NATIVE_SPEECH_OUTPUT_THRESHOLD ? "automatic-above-200b" : "automatic-through-200b",
    available: outputAvailable,
    detail: outputRoute === null ? "Automatic output is waiting for verified native parameter accounting; no output route was assumed."
      : outputAvailable ? outputRoute === "native"
        ? "Same-brain waveform output is selected. Speech quality still depends on learned audio capability and is not guaranteed by parameter count."
        : "Platform text-to-speech is selected."
      : outputRoute === "native" ? "Requested same-brain waveform output or playback is unavailable; platform text-to-speech was not substituted."
        : "Platform text-to-speech is unavailable; native output was not substituted." };
  return { input, output, ready: inputAvailable && outputAvailable,
    controllerPreferences: outputRoute === null ? null : {
      schemaVersion: 1, deliveryMode: selected.deliveryMode, pace: selected.pace,
      neuralListening: inputRoute === "native", neuralVoice: outputRoute === "native"
    } };
}
