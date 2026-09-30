import type { ChatResult, ChatStreamEvent } from "./types";

export const LIVE_VOICE_SCHEMA_VERSION = 1 as const;
export const LIVE_VOICE_PREFERENCES_SCHEMA_VERSION = 1 as const;

export type LiveVoiceDeliveryMode = "live" | "buffered";
export type LiveVoicePace = "slow" | "normal" | "fast";

/** User-owned audio presentation choices; these never enter neural state. */
export interface LiveVoicePreferences {
  schemaVersion: typeof LIVE_VOICE_PREFERENCES_SCHEMA_VERSION;
  deliveryMode: LiveVoiceDeliveryMode;
  pace: LiveVoicePace;
  neuralListening: boolean;
  neuralVoice: boolean;
}

export type LiveVoicePhase =
  | "idle"
  | "starting"
  | "listening"
  | "pondering"
  | "speaking"
  | "rendering-audio"
  | "stopping"
  | "unavailable"
  | "error";

export type LiveVoiceErrorCode =
  | "microphone-unavailable"
  | "permission-denied"
  | "permission-timeout"
  | "no-input-device"
  | "device-busy"
  | "capture-failed"
  | "recognition-unavailable"
  | "neural-listening-unavailable"
  | "neural-voice-unavailable"
  | "recognition-failed"
  | "synthesis-failed"
  | "chat-failed";

export interface LiveVoiceCapabilities {
  microphoneCapture: boolean;
  speechRecognition: boolean;
  speechSynthesis: boolean;
  /** Recognition remains active while a reply is being spoken. */
  fullDuplex: boolean;
  /** A speech-start event can synchronously stop the local speech queue. */
  bargeIn: boolean;
  /** Direct same-brain paths; never aliases for platform STT/TTS. */
  neuralListening: boolean;
  neuralVoice: boolean;
  neuralVoiceQuality?: "needs-speech-training" | "unverified";
  recognitionDetail?: string;
  neuralDetail?: string;
}

export interface LiveVoiceError {
  code: LiveVoiceErrorCode;
  message: string;
  recoverable: boolean;
}

/**
 * One recognition-finalized chunk. The ID also scopes the chat turn so a late
 * reply from an interrupted utterance can never become the active voice reply.
 */
export interface LiveVoiceUtterance {
  id: string;
  turnId: string;
  transcript: string;
  confidence?: number;
  sequence: number;
  createdAt: string;
  deliveryMode: LiveVoiceDeliveryMode;
  pace: LiveVoicePace;
}

export interface LiveVoiceReply {
  utteranceId: string;
  turnId: string;
  text: string;
  acceptedAt: string;
  /** The authoritative chat result is forwarded only after correlation wins. */
  chatResult?: ChatResult;
}

export interface LiveVoiceState {
  schemaVersion: typeof LIVE_VOICE_SCHEMA_VERSION;
  enabled: boolean;
  phase: LiveVoicePhase;
  capabilities: LiveVoiceCapabilities;
  preferences: LiveVoicePreferences;
  capture: "idle" | "requesting" | "active" | "error";
  recognition: "unavailable" | "idle" | "listening" | "error";
  synthesis: "unavailable" | "idle" | "preparing" | "speaking" | "error";
  interimTranscript: string;
  activeUtterance?: LiveVoiceUtterance;
  lastUtterance?: LiveVoiceUtterance;
  lastReply?: LiveVoiceReply;
  error?: LiveVoiceError;
  speechBackpressure: {
    queuedChunks: number;
    pendingCharacters: number;
    skippedCharacters: number;
    maxQueuedChunks: number;
  };
  /** Voice sends the recognized words verbatim; it never manufactures a prompt. */
  hiddenBehavioralPrompt: false;
}

export interface LiveVoiceRecognitionFinal {
  transcript: string;
  confidence?: number;
}

export interface LiveVoiceRecognitionError {
  /** Web Speech error name where applicable (for example, `not-allowed`). */
  code: string;
  message?: string;
}

export interface LiveVoiceRecognitionHandlers {
  onSpeechStart(): void;
  onInterim(transcript: string): void;
  onFinal(result: LiveVoiceRecognitionFinal): void;
  onError(error: LiveVoiceRecognitionError): void;
  onEnd(): void;
}

export interface LiveVoiceCaptureSession {
  stop(): void;
}

export interface LiveVoiceCaptureAdapter {
  readonly available: boolean;
  /** This is the only call allowed to trigger the OS microphone prompt. */
  request(): Promise<LiveVoiceCaptureSession>;
}

export interface LiveVoiceNeuralCapabilityStatus {
  /** Raw microphone audio can enter trained assemblies/STDP in this brain. */
  listening: boolean;
  /** Same-brain waveform rendering is connected; intelligibility is separate. */
  voice: boolean;
  voiceQuality?: "needs-speech-training" | "unverified";
  detail: string;
}

/**
 * Optional direct neural audio path. Listening is a same-brain sensory stream,
 * not an alias for speech recognition. A platform transcript may still be
 * used at the language boundary. Neural speech output is exposed only when an
 * adapter connects the same brain audio region; unverified quality is explicit.
 */
export interface LiveVoiceNeuralAdapter {
  inspect(brainId: string): Promise<LiveVoiceNeuralCapabilityStatus>;
  startListening(brainId: string): Promise<LiveVoiceCaptureSession>;
  synthesis?: LiveVoiceSynthesisAdapter;
}

export interface LiveVoiceRecognitionSession {
  stop(): void;
  abort(): void;
}

export interface LiveVoiceRecognitionAdapter {
  readonly available: boolean;
  readonly detail?: string;
  start(handlers: LiveVoiceRecognitionHandlers):
    | LiveVoiceRecognitionSession
    | Promise<LiveVoiceRecognitionSession>;
}

export interface LiveVoiceSynthesisSession {
  cancel(): void;
}

export interface LiveVoiceSynthesisHandlers {
  onStart(): void;
  onEnd(): void;
  onError(message?: string): void;
}

export interface LiveVoiceSynthesisAdapter {
  readonly available: boolean;
  readonly deferredStart?: boolean;
  speak(
    text: string,
    handlers: LiveVoiceSynthesisHandlers,
    options: { rate: number }
  ): LiveVoiceSynthesisSession;
  cancel(): void;
}

export interface LiveVoiceChatAdapter {
  send(brainId: string, transcript: string, turnId: string): Promise<ChatResult>;
  cancel(brainId: string, turnId: string): Promise<number>;
  subscribe?(listener: (event: ChatStreamEvent) => void): () => void;
}
