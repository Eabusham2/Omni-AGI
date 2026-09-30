import type {
  LiveVoiceCapabilities,
  LiveVoiceCaptureAdapter,
  LiveVoiceCaptureSession,
  LiveVoiceChatAdapter,
  LiveVoiceError,
  LiveVoiceNeuralAdapter,
  LiveVoicePreferences,
  LiveVoiceRecognitionAdapter,
  LiveVoiceRecognitionError,
  LiveVoiceRecognitionFinal,
  LiveVoiceRecognitionSession,
  LiveVoiceReply,
  LiveVoiceState,
  LiveVoiceSynthesisAdapter,
  LiveVoiceSynthesisSession,
  LiveVoiceUtterance
} from "../../shared/liveVoice";
import {
  LIVE_VOICE_PREFERENCES_SCHEMA_VERSION,
  LIVE_VOICE_SCHEMA_VERSION
} from "../../shared/liveVoice";
import type { ChatStreamEvent } from "../../shared/types";

export interface LiveVoiceControllerOptions {
  brainId: string;
  capture: LiveVoiceCaptureAdapter;
  recognition: LiveVoiceRecognitionAdapter;
  synthesis: LiveVoiceSynthesisAdapter;
  neural?: LiveVoiceNeuralAdapter;
  chat: LiveVoiceChatAdapter;
  preferences?: LiveVoicePreferences;
  /** At most this many not-yet-spoken chunks are retained. */
  maxLiveSpeechQueue?: number;
  /** Maximum characters per incremental speech unit. */
  liveSpeechChunkCharacters?: number;
  createId?: () => string;
  now?: () => Date;
  /** Injected by tests; production uses a short restart delay. */
  scheduleRecognitionRestart?: (callback: () => void) => void;
  onAcceptedReply?: (reply: LiveVoiceReply) => void;
}

export type LiveVoiceStateListener = (state: Readonly<LiveVoiceState>) => void;

export const LIVE_VOICE_PACE_RATES = {
  slow: 0.8,
  normal: 1,
  fast: 1.35
} as const;

const defaultPreferences: LiveVoicePreferences = {
  schemaVersion: LIVE_VOICE_PREFERENCES_SCHEMA_VERSION,
  deliveryMode: "live",
  pace: "normal",
  neuralListening: false,
  neuralVoice: false
};

const readableError = (error: unknown): string =>
  error instanceof Error ? error.message : String(error ?? "Unknown voice error");

function captureError(error: unknown): LiveVoiceError {
  const name =
    typeof error === "object" && error !== null && "name" in error
      ? String((error as { name: unknown }).name)
      : "";
  const message = readableError(error);
  if (name === "NotAllowedError" || name === "SecurityError") {
    return {
      code: "permission-denied",
      message: "Microphone access was not granted. Enable it in system or app privacy settings and try again.",
      recoverable: true
    };
  }
  if (name === "TimeoutError") {
    return {
      code: "permission-timeout",
      message:
        "Microphone permission did not respond. Check the operating system's microphone privacy settings. On macOS development runs, allow Electron (packaged runs use Omni AGI Studio), restart the app, and try again.",
      recoverable: true
    };
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return {
      code: "no-input-device",
      message: "No microphone input device is available.",
      recoverable: true
    };
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return {
      code: "device-busy",
      message: "The microphone could not be opened; another application may be using it.",
      recoverable: true
    };
  }
  return {
    code: "capture-failed",
    message: `Microphone capture failed: ${message}`,
    recoverable: true
  };
}

function recognitionError(error: LiveVoiceRecognitionError): LiveVoiceError {
  switch (error.code) {
    case "not-allowed":
    case "service-not-allowed":
      return {
        code: "permission-denied",
        message: "Speech recognition was not granted microphone access.",
        recoverable: true
      };
    case "audio-capture":
      return {
        code: "no-input-device",
        message: "Speech recognition cannot access an audio input device.",
        recoverable: true
      };
    default:
      return {
        code: "recognition-failed",
        message: error.message
          ? `Speech recognition failed: ${error.message}`
          : `Speech recognition failed (${error.code}).`,
        recoverable: true
      };
  }
}

function unavailableError(
  code: "microphone-unavailable" | "recognition-unavailable",
  detail?: string
): LiveVoiceError {
  return code === "microphone-unavailable"
    ? {
        code,
        message: "This runtime cannot request microphone capture.",
        recoverable: false
      }
    : {
        code,
        message:
          detail ??
          "Speech recognition is unavailable in this Electron/Chromium runtime. No transcript will be simulated.",
        recoverable: false
      };
}

/**
 * Transport-neutral live voice state machine. Capture, ASR, synthesis, and
 * chat are injected so the controller has no hidden prompt or platform magic.
 */
export class LiveVoiceController {
  private state: LiveVoiceState;
  private readonly listeners = new Set<LiveVoiceStateListener>();
  private captureSession?: LiveVoiceCaptureSession;
  private recognitionSession?: LiveVoiceRecognitionSession;
  private synthesisSession?: LiveVoiceSynthesisSession;
  private outputSynthesis: LiveVoiceSynthesisAdapter;
  private neuralCapabilityProbe?: Promise<void>;
  private lifecycle = 0;
  private recognitionRun = 0;
  private replyGeneration = 0;
  private utteranceSequence = 0;
  private disposed = false;
  private readonly removeChatStream?: () => void;
  private liveSpeechQueue: string[] = [];
  private liveSpeechBuffer = "";
  private liveStreamText = "";
  private liveChatComplete = false;
  private liveSkippedCharacters = 0;
  private liveLastSequence = -1;
  private bufferedRequiresFreshSpeechStart = false;

  constructor(private readonly options: LiveVoiceControllerOptions) {
    this.outputSynthesis = options.synthesis;
    const capabilities: LiveVoiceCapabilities = {
      microphoneCapture: options.capture.available,
      speechRecognition: options.recognition.available,
      speechSynthesis: options.synthesis.available,
      fullDuplex: options.capture.available && options.recognition.available,
      bargeIn: options.synthesis.available,
      neuralListening: false,
      neuralVoice: false,
      recognitionDetail: options.recognition.detail,
      neuralDetail:
        options.neural
          ? "Checking this brain's trained neural audio capabilities. Platform STT/TTS remain separate."
          : "No direct neural audio adapter is installed. Platform speech remains clearly separate."
    };
    this.state = {
      schemaVersion: LIVE_VOICE_SCHEMA_VERSION,
      enabled: false,
      phase: "idle",
      capabilities,
      preferences: options.preferences ?? defaultPreferences,
      capture: "idle",
      recognition: options.recognition.available ? "idle" : "unavailable",
      synthesis: options.synthesis.available ? "idle" : "unavailable",
      interimTranscript: "",
      speechBackpressure: {
        queuedChunks: 0,
        pendingCharacters: 0,
        skippedCharacters: 0,
        maxQueuedChunks: this.maxLiveSpeechQueue()
      },
      hiddenBehavioralPrompt: false
    };
    this.removeChatStream = options.chat.subscribe?.((event) =>
      this.handleChatStream(event)
    );
    if (options.neural) {
      this.neuralCapabilityProbe = this.refreshNeuralCapabilities();
    }
  }

  getState(): Readonly<LiveVoiceState> {
    return this.state;
  }

  subscribe(listener: LiveVoiceStateListener): () => void {
    this.listeners.add(listener);
    listener(this.state);
    return () => this.listeners.delete(listener);
  }

  setPreferences(preferences: LiveVoicePreferences): void {
    if (
      preferences.schemaVersion !== LIVE_VOICE_PREFERENCES_SCHEMA_VERSION ||
      !["live", "buffered"].includes(preferences.deliveryMode) ||
      !["slow", "normal", "fast"].includes(preferences.pace) ||
      typeof preferences.neuralListening !== "boolean" ||
      typeof preferences.neuralVoice !== "boolean"
    ) {
      throw new Error("Invalid live voice presentation preferences.");
    }
    this.patch({ preferences: { ...preferences } });
  }

  private async refreshNeuralCapabilities(): Promise<void> {
    const neural = this.options.neural;
    if (!neural) return;
    try {
      const measured = await neural.inspect(this.options.brainId);
      if (this.disposed) return;
      const listening = measured.listening === true;
      const voice =
        measured.voice === true && neural.synthesis?.available === true;
      this.patch({
        capabilities: {
          ...this.state.capabilities,
          neuralListening: listening,
          neuralVoice: voice,
          neuralVoiceQuality: measured.voiceQuality,
          bargeIn:
            this.options.synthesis.available ||
            (voice && neural.synthesis?.available === true),
          neuralDetail: measured.detail
        }
      });
    } catch (error) {
      if (this.disposed) return;
      this.patch({
        capabilities: {
          ...this.state.capabilities,
          neuralListening: false,
          neuralVoice: false,
          neuralDetail: `Neural audio readiness could not be verified: ${readableError(error)} Platform STT/TTS remain separate.`
        }
      });
    }
  }

  async start(): Promise<void> {
    if (this.disposed) throw new Error("Live voice controller has been disposed.");
    if (this.state.enabled && this.state.phase !== "error") return;
    if (
      this.state.preferences.neuralListening ||
      this.state.preferences.neuralVoice
    ) {
      await this.neuralCapabilityProbe;
    }
    this.bufferedRequiresFreshSpeechStart = false;
    if (
      this.state.preferences.neuralListening &&
      !this.state.capabilities.neuralListening
    ) {
      this.patch({
        enabled: false,
        phase: "unavailable",
        error: {
          code: "neural-listening-unavailable",
          message:
            "Neural listening was requested, but no compatible trained audio-input pack is installed. Platform speech recognition was not substituted.",
          recoverable: true
        }
      });
      return;
    }
    if (this.state.preferences.neuralVoice && !this.state.capabilities.neuralVoice) {
      this.patch({
        enabled: false,
        phase: "unavailable",
        error: {
          code: "neural-voice-unavailable",
          message:
            "Own waveform output or local playback is unavailable. Platform speech synthesis was not substituted.",
          recoverable: true
        }
      });
      return;
    }
    this.outputSynthesis = this.state.preferences.neuralVoice
      ? this.options.neural?.synthesis ?? this.options.synthesis
      : this.options.synthesis;
    if (!this.options.capture.available) {
      this.patch({
        enabled: false,
        phase: "unavailable",
        capture: "error",
        error: unavailableError("microphone-unavailable")
      });
      return;
    }
    if (!this.options.recognition.available) {
      this.patch({
        enabled: false,
        phase: "unavailable",
        recognition: "unavailable",
        error: unavailableError(
          "recognition-unavailable",
          this.options.recognition.detail
        )
      });
      return;
    }

    const lifecycle = ++this.lifecycle;
    this.patch({
      enabled: true,
      phase: "starting",
      capture: "requesting",
      recognition: "idle",
      synthesis: this.outputSynthesis.available ? "idle" : "unavailable",
      interimTranscript: "",
      activeUtterance: undefined,
      error: undefined
    });
    try {
      const session = this.state.preferences.neuralListening
        ? await this.options.neural!.startListening(this.options.brainId)
        : await this.options.capture.request();
      if (!this.isCurrent(lifecycle)) {
        session.stop();
        return;
      }
      this.captureSession = session;
      this.patch({ capture: "active" });
      await this.startRecognition(lifecycle);
    } catch (error) {
      if (!this.isCurrent(lifecycle)) return;
      const mapped = captureError(error);
      await this.fail(
        this.state.preferences.neuralListening && mapped.code === "capture-failed"
          ? {
              code: "neural-listening-unavailable",
              message: `Direct neural listening could not start: ${readableError(error)} Platform speech recognition was not substituted.`,
              recoverable: true
            }
          : mapped
      );
    }
  }

  /** Stops capture, ASR, speech, and the exactly correlated chat turn. */
  async stop(): Promise<void> {
    if (this.disposed && this.state.phase === "idle") return;
    ++this.lifecycle;
    ++this.replyGeneration;
    this.bufferedRequiresFreshSpeechStart = false;
    const turnId = this.state.activeUtterance?.turnId;
    this.patch({ enabled: false, phase: "stopping", interimTranscript: "" });
    this.cancelSynthesis();
    this.abortRecognition();
    this.stopCapture();

    let cancellationError: LiveVoiceError | undefined;
    if (turnId) {
      try {
        await this.options.chat.cancel(this.options.brainId, turnId);
      } catch (error) {
        cancellationError = {
          code: "chat-failed",
          message: `The correlated chat turn could not be cancelled: ${readableError(error)}`,
          recoverable: true
        };
      }
    }
    this.patch({
      enabled: false,
      phase: "idle",
      capture: "idle",
      recognition: this.options.recognition.available ? "idle" : "unavailable",
      synthesis: this.outputSynthesis.available ? "idle" : "unavailable",
      activeUtterance: undefined,
      error: cancellationError
    });
  }

  /**
   * Public barge-in hook for push-to-talk UI. Recognition speech-start events
   * call the same synchronous cancellation path automatically.
   */
  interrupt(): void {
    if (!this.state.enabled) return;
    this.handleSpeechStart(this.lifecycle);
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    await this.stop();
    this.disposed = true;
    this.removeChatStream?.();
    this.listeners.clear();
  }

  private async startRecognition(lifecycle: number): Promise<void> {
    if (!this.isCurrent(lifecycle)) return;
    const run = ++this.recognitionRun;
    this.patch({
      phase: "listening",
      recognition: "listening",
      error: undefined
    });
    try {
      const session = await this.options.recognition.start({
        onSpeechStart: () => this.handleSpeechStart(lifecycle),
        onInterim: (transcript) => this.handleInterim(lifecycle, transcript),
        onFinal: (result) => void this.handleFinal(lifecycle, result),
        onError: (error) => void this.handleRecognitionError(lifecycle, error),
        onEnd: () => this.handleRecognitionEnd(lifecycle, run)
      });
      if (!this.isCurrent(lifecycle) || run !== this.recognitionRun) {
        session.abort();
        return;
      }
      this.recognitionSession = session;
    } catch (error) {
      if (!this.isCurrent(lifecycle)) return;
      await this.fail({
        code: "recognition-failed",
        message: `Speech recognition could not start: ${readableError(error)}`,
        recoverable: true
      });
    }
  }

  private handleSpeechStart(lifecycle: number): void {
    if (!this.isCurrent(lifecycle)) return;
    const interrupted = this.state.activeUtterance;
    // Buffered delivery deliberately ignores microphone-triggered barge-in.
    // The visible Stop control remains available and calls stop() directly.
    if (interrupted?.deliveryMode === "buffered") return;
    if (!interrupted) this.bufferedRequiresFreshSpeechStart = false;
    ++this.replyGeneration;

    // These calls happen before any asynchronous work: audible speech stops
    // immediately and the exact chat correlation is cancelled immediately.
    this.cancelSynthesis();
    if (interrupted) {
      void this.options.chat
        .cancel(this.options.brainId, interrupted.turnId)
        .catch(() => undefined);
    }
    this.patch({
      phase: "listening",
      synthesis: this.outputSynthesis.available ? "idle" : "unavailable",
      interimTranscript: "",
      activeUtterance: undefined,
      lastUtterance: interrupted ?? this.state.lastUtterance,
      error: undefined
    });
  }

  private handleInterim(lifecycle: number, transcript: string): void {
    if (!this.isCurrent(lifecycle)) return;
    // A continuous recognizer may publish an interim fragment in the same
    // callback batch as a finalized result. It must not demote the finalized
    // utterance's correlated Ponder turn back to listening.
    if (this.state.activeUtterance) return;
    // Buffered speech ignores recognition while TTS is active. Once the
    // recognizer observes genuinely fresh speech, an interim fragment is also
    // sufficient to re-arm engines that omit a speech-start callback.
    if (this.bufferedRequiresFreshSpeechStart && transcript.trim()) {
      this.bufferedRequiresFreshSpeechStart = false;
    }
    this.patch({
      phase: "listening",
      interimTranscript: transcript.replace(/\s+/g, " ").trim()
    });
  }

  private async handleFinal(
    lifecycle: number,
    result: LiveVoiceRecognitionFinal
  ): Promise<void> {
    if (!this.isCurrent(lifecycle)) return;
    const transcript = result.transcript.replace(/\s+/g, " ").trim();
    if (this.bufferedRequiresFreshSpeechStart) return;
    if (!transcript) {
      this.patch({ interimTranscript: "" });
      return;
    }

    // Some recognition engines omit speech-start. A final chunk still creates
    // a new fork and invalidates every older reply before it is sent.
    const previous = this.state.activeUtterance;
    if (previous?.deliveryMode === "buffered") {
      // Recognition may hear the buffered TTS output. Do not turn it into a
      // second user message or cancel the buffered reply.
      this.patch({ interimTranscript: "" });
      return;
    }
    if (previous) {
      ++this.replyGeneration;
      this.cancelSynthesis();
      void this.options.chat
        .cancel(this.options.brainId, previous.turnId)
        .catch(() => undefined);
    }
    const utteranceId = this.options.createId?.() ?? crypto.randomUUID();
    const utterance: LiveVoiceUtterance = {
      id: utteranceId,
      turnId: `voice-${utteranceId}`,
      transcript,
      confidence:
        typeof result.confidence === "number" && Number.isFinite(result.confidence)
          ? Math.max(0, Math.min(1, result.confidence))
          : undefined,
      sequence: ++this.utteranceSequence,
      createdAt: (this.options.now?.() ?? new Date()).toISOString(),
      deliveryMode: this.state.preferences.deliveryMode,
      pace: this.state.preferences.pace
    };
    this.resetLiveSpeech();
    const replyGeneration = ++this.replyGeneration;
    this.patch({
      phase: "pondering",
      interimTranscript: "",
      activeUtterance: utterance,
      lastUtterance: previous ?? this.state.lastUtterance,
      error: undefined
    });

    try {
      // The recognized transcript is sent verbatim. No system/persona text,
      // command prefix, or behavioral instruction exists in this path.
      const result = await this.options.chat.send(
        this.options.brainId,
        transcript,
        utterance.turnId
      );
      if (!this.acceptsReply(lifecycle, replyGeneration, utterance.id)) return;
      const text = result.brainMessage.content;
      const reply: LiveVoiceReply = {
        utteranceId: utterance.id,
        turnId: utterance.turnId,
        text,
        acceptedAt: (this.options.now?.() ?? new Date()).toISOString(),
        chatResult: result
      };
      this.patch({ lastReply: reply, error: undefined });
      try {
        this.options.onAcceptedReply?.(reply);
      } catch {
        // Presentation callbacks cannot corrupt the audio state machine.
      }
      if (!text.trim() || !this.outputSynthesis.available) {
        this.resetLiveSpeech();
        this.patch({
          phase: "listening",
          synthesis: this.outputSynthesis.available ? "idle" : "unavailable",
          activeUtterance: undefined,
          lastUtterance: utterance
        });
        return;
      }
      if (utterance.deliveryMode === "buffered") {
        this.speakBufferedReply(lifecycle, replyGeneration, utterance, text);
        return;
      }
      this.liveChatComplete = true;
      if (!this.liveStreamText) {
        this.feedLiveSpeech(text);
      } else if (text.startsWith(this.liveStreamText)) {
        this.feedLiveSpeech(text.slice(this.liveStreamText.length));
      }
      this.extractLiveSpeechChunks(true);
      this.drainLiveSpeech(lifecycle, replyGeneration, utterance);
    } catch (error) {
      if (!this.acceptsReply(lifecycle, replyGeneration, utterance.id)) return;
      this.patch({
        phase: "listening",
        activeUtterance: undefined,
        lastUtterance: utterance,
        error: {
          code: "chat-failed",
          message: `Voice chat failed: ${readableError(error)}`,
          recoverable: true
        }
      });
    }
  }

  private speakBufferedReply(
    lifecycle: number,
    replyGeneration: number,
    utterance: LiveVoiceUtterance,
    text: string
  ): void {
    try {
      const session = this.outputSynthesis.speak(
        text,
        {
          onStart: () => {
            if (!this.acceptsReply(lifecycle, replyGeneration, utterance.id)) return;
            this.patch({ phase: "speaking", synthesis: "speaking" });
          },
          onEnd: () => {
            if (!this.acceptsReply(lifecycle, replyGeneration, utterance.id)) return;
            this.synthesisSession = undefined;
            this.bufferedRequiresFreshSpeechStart = true;
            this.patch({
              phase: "listening",
              synthesis: "idle",
              activeUtterance: undefined,
              lastUtterance: utterance
            });
          },
          onError: (message) => {
            if (!this.acceptsReply(lifecycle, replyGeneration, utterance.id)) return;
            this.synthesisSession = undefined;
            this.patch({
              phase: "listening",
              synthesis: "error",
              activeUtterance: undefined,
              lastUtterance: utterance,
              error: {
                code: "synthesis-failed",
                message: message
                  ? `Speech synthesis failed: ${message}`
                  : "Speech synthesis failed.",
                recoverable: true
              }
            });
          }
        },
        { rate: LIVE_VOICE_PACE_RATES[utterance.pace] }
      );
      if (!this.acceptsReply(lifecycle, replyGeneration, utterance.id)) {
        session.cancel();
        return;
      }
      this.synthesisSession = session;
      // Some adapters do not publish a distinct start event.
      this.patch(this.outputSynthesis.deferredStart
        ? { phase: "rendering-audio", synthesis: "preparing" }
        : { phase: "speaking", synthesis: "speaking" });
    } catch (error) {
      if (!this.acceptsReply(lifecycle, replyGeneration, utterance.id)) return;
      this.patch({
        phase: "listening",
        synthesis: "error",
        activeUtterance: undefined,
        lastUtterance: utterance,
        error: {
          code: "synthesis-failed",
          message: `Speech synthesis failed: ${readableError(error)}`,
          recoverable: true
        }
      });
    }
  }

  private handleChatStream(event: ChatStreamEvent): void {
    const utterance = this.state.activeUtterance;
    if (
      !utterance ||
      utterance.deliveryMode !== "live" ||
      event.brainId !== this.options.brainId ||
      event.turnId !== utterance.turnId ||
      event.sequence <= this.liveLastSequence
    ) {
      return;
    }
    this.liveLastSequence = event.sequence;
    if (event.type !== "chat-token" || !event.delta) return;
    this.liveStreamText += event.delta;
    this.feedLiveSpeech(event.delta);
    this.extractLiveSpeechChunks(false);
    this.drainLiveSpeech(this.lifecycle, this.replyGeneration, utterance);
  }

  private feedLiveSpeech(text: string): void {
    if (!text) return;
    this.liveSpeechBuffer += text;
  }

  private extractLiveSpeechChunks(flush: boolean): void {
    const limit = this.liveSpeechChunkCharacters();
    while (this.liveSpeechBuffer) {
      const boundary = this.speechBoundary(this.liveSpeechBuffer, limit, flush);
      if (boundary <= 0) break;
      const chunk = this.liveSpeechBuffer.slice(0, boundary).trim();
      this.liveSpeechBuffer = this.liveSpeechBuffer.slice(boundary).trimStart();
      if (chunk) this.enqueueLiveSpeech(chunk);
      if (!flush && this.liveSpeechBuffer.length < limit) break;
    }
    this.publishBackpressure();
  }

  private speechBoundary(text: string, limit: number, flush: boolean): number {
    const bounded = text.slice(0, limit);
    const sentenceMatches = [...bounded.matchAll(/[.!?](?:[\s\n]+|$)/g)];
    const sentence = sentenceMatches.at(-1);
    if (sentence?.index !== undefined && sentence.index >= 20) {
      return sentence.index + sentence[0].length;
    }
    if (text.length >= limit) {
      const whitespace = Math.max(
        bounded.lastIndexOf(" "),
        bounded.lastIndexOf("\n")
      );
      return whitespace >= Math.floor(limit * 0.55) ? whitespace + 1 : limit;
    }
    return flush ? text.length : 0;
  }

  private enqueueLiveSpeech(chunk: string): void {
    const maximum = this.maxLiveSpeechQueue();
    if (this.liveSpeechQueue.length < maximum) {
      this.liveSpeechQueue.push(chunk);
      this.publishBackpressure();
      return;
    }
    // Generation may outrun speech, but spoken content must not disappear.
    // Coalesce overflow into the final not-yet-native unit; drainLiveSpeech
    // slices that unit back into bounded utterances. This bounds the browser's
    // native speech queue without skipping any accepted model text.
    this.liveSpeechQueue[maximum - 1] = [
      this.liveSpeechQueue[maximum - 1],
      chunk
    ].filter(Boolean).join(" ");
    this.publishBackpressure();
  }

  private drainLiveSpeech(
    lifecycle: number,
    replyGeneration: number,
    utterance: LiveVoiceUtterance
  ): void {
    if (
      !this.acceptsReply(lifecycle, replyGeneration, utterance.id) ||
      utterance.deliveryMode !== "live" ||
      this.synthesisSession
    ) {
      return;
    }
    const pending = this.liveSpeechQueue.shift();
    let chunk = pending;
    if (pending && pending.length > this.liveSpeechChunkCharacters()) {
      const boundary = this.speechBoundary(
        pending,
        this.liveSpeechChunkCharacters(),
        true
      );
      chunk = pending.slice(0, boundary).trim();
      const remainder = pending.slice(boundary).trimStart();
      if (remainder) this.liveSpeechQueue.unshift(remainder);
    }
    this.publishBackpressure();
    if (!chunk) {
      if (this.liveChatComplete && !this.liveSpeechBuffer) {
        this.finishLiveSpeech(utterance);
      } else {
        this.patch({ phase: "pondering", synthesis: "idle" });
      }
      return;
    }
    try {
      const session = this.outputSynthesis.speak(
        chunk,
        {
          onStart: () => {
            if (!this.acceptsReply(lifecycle, replyGeneration, utterance.id)) return;
            this.patch({ phase: "speaking", synthesis: "speaking" });
          },
          onEnd: () => {
            if (!this.acceptsReply(lifecycle, replyGeneration, utterance.id)) return;
            this.synthesisSession = undefined;
            this.patch({ synthesis: "idle" });
            this.drainLiveSpeech(lifecycle, replyGeneration, utterance);
          },
          onError: (message) => {
            if (!this.acceptsReply(lifecycle, replyGeneration, utterance.id)) return;
            this.synthesisSession = undefined;
            this.liveSpeechQueue = [];
            this.liveSpeechBuffer = "";
            this.publishBackpressure();
            this.patch({
              phase: "listening",
              synthesis: "error",
              activeUtterance: undefined,
              lastUtterance: utterance,
              error: {
                code: "synthesis-failed",
                message: message
                  ? `Speech synthesis failed: ${message}`
                  : "Speech synthesis failed.",
                recoverable: true
              }
            });
          }
        },
        { rate: LIVE_VOICE_PACE_RATES[utterance.pace] }
      );
      if (!this.acceptsReply(lifecycle, replyGeneration, utterance.id)) {
        session.cancel();
        return;
      }
      this.synthesisSession = session;
      this.patch(this.outputSynthesis.deferredStart
        ? { phase: "rendering-audio", synthesis: "preparing" }
        : { phase: "speaking", synthesis: "speaking" });
    } catch (error) {
      if (!this.acceptsReply(lifecycle, replyGeneration, utterance.id)) return;
      this.liveSpeechQueue = [];
      this.liveSpeechBuffer = "";
      this.publishBackpressure();
      this.patch({
        phase: "listening",
        synthesis: "error",
        activeUtterance: undefined,
        lastUtterance: utterance,
        error: {
          code: "synthesis-failed",
          message: `Speech synthesis failed: ${readableError(error)}`,
          recoverable: true
        }
      });
    }
  }

  private finishLiveSpeech(utterance: LiveVoiceUtterance): void {
    this.synthesisSession = undefined;
    this.patch({
      phase: "listening",
      synthesis: "idle",
      activeUtterance: undefined,
      lastUtterance: utterance
    });
  }

  private maxLiveSpeechQueue(): number {
    return Math.max(1, Math.min(8, this.options.maxLiveSpeechQueue ?? 3));
  }

  private liveSpeechChunkCharacters(): number {
    return Math.max(
      48,
      Math.min(480, this.options.liveSpeechChunkCharacters ?? 180)
    );
  }

  private publishBackpressure(): void {
    this.patch({
      speechBackpressure: {
        queuedChunks: this.liveSpeechQueue.length,
        pendingCharacters:
          this.liveSpeechBuffer.length +
          this.liveSpeechQueue.reduce((sum, chunk) => sum + chunk.length, 0),
        skippedCharacters: this.liveSkippedCharacters,
        maxQueuedChunks: this.maxLiveSpeechQueue()
      }
    });
  }

  private resetLiveSpeech(): void {
    this.liveSpeechQueue = [];
    this.liveSpeechBuffer = "";
    this.liveStreamText = "";
    this.liveChatComplete = false;
    this.liveSkippedCharacters = 0;
    this.liveLastSequence = -1;
    this.patch({
      speechBackpressure: {
        queuedChunks: 0,
        pendingCharacters: 0,
        skippedCharacters: 0,
        maxQueuedChunks: this.maxLiveSpeechQueue()
      }
    });
  }

  private async handleRecognitionError(
    lifecycle: number,
    error: LiveVoiceRecognitionError
  ): Promise<void> {
    if (!this.isCurrent(lifecycle) || error.code === "aborted") return;
    if (error.code === "no-speech") {
      this.patch({
        recognition: "listening",
        error: {
          code: "recognition-failed",
          message: "No speech was detected; listening continues.",
          recoverable: true
        }
      });
      return;
    }
    await this.fail(recognitionError(error));
  }

  private handleRecognitionEnd(lifecycle: number, run: number): void {
    if (!this.isCurrent(lifecycle) || run !== this.recognitionRun) return;
    this.recognitionSession = undefined;
    this.patch({ recognition: "idle" });
    const schedule =
      this.options.scheduleRecognitionRestart ??
      ((callback: () => void) => globalThis.setTimeout(callback, 250));
    schedule(() => {
      if (!this.isCurrent(lifecycle) || this.recognitionSession) return;
      void this.startRecognition(lifecycle);
    });
  }

  private async fail(error: LiveVoiceError): Promise<void> {
    ++this.lifecycle;
    ++this.replyGeneration;
    const turnId = this.state.activeUtterance?.turnId;
    this.cancelSynthesis();
    this.abortRecognition();
    this.stopCapture();
    if (turnId) {
      void this.options.chat
        .cancel(this.options.brainId, turnId)
        .catch(() => undefined);
    }
    this.patch({
      enabled: false,
      phase: "error",
      capture: "error",
      recognition: this.options.recognition.available ? "error" : "unavailable",
      synthesis: this.outputSynthesis.available ? "idle" : "unavailable",
      activeUtterance: undefined,
      interimTranscript: "",
      error
    });
  }

  private cancelSynthesis(): void {
    const session = this.synthesisSession;
    this.synthesisSession = undefined;
    try {
      session?.cancel();
    } catch {
      // A cancelled or removed output device is already silent.
    }
    try {
      this.outputSynthesis.cancel();
    } catch {
      // Preserve barge-in and continue listening if the platform queue vanished.
    }
    this.liveSpeechQueue = [];
    this.liveSpeechBuffer = "";
    this.liveStreamText = "";
    this.liveChatComplete = false;
    this.liveSkippedCharacters = 0;
    this.liveLastSequence = -1;
    this.publishBackpressure();
  }

  private abortRecognition(): void {
    const session = this.recognitionSession;
    this.recognitionSession = undefined;
    ++this.recognitionRun;
    try {
      session?.abort();
    } catch {
      // The session may already have ended itself.
    }
  }

  private stopCapture(): void {
    const session = this.captureSession;
    this.captureSession = undefined;
    try {
      session?.stop();
    } catch {
      // Tracks that ended due to device removal need no further cleanup.
    }
  }

  private acceptsReply(
    lifecycle: number,
    replyGeneration: number,
    utteranceId: string
  ): boolean {
    return (
      this.isCurrent(lifecycle) &&
      replyGeneration === this.replyGeneration &&
      this.state.activeUtterance?.id === utteranceId
    );
  }

  private isCurrent(lifecycle: number): boolean {
    return !this.disposed && this.state.enabled && lifecycle === this.lifecycle;
  }

  private patch(patch: Partial<LiveVoiceState>): void {
    this.state = { ...this.state, ...patch };
    for (const listener of this.listeners) listener(this.state);
  }
}
