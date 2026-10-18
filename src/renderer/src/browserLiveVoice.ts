import type {
  LiveVoiceCapabilities,
  LiveVoiceCaptureAdapter,
  LiveVoiceChatAdapter,
  LiveVoiceNeuralAdapter,
  LiveVoiceRecognitionAdapter,
  LiveVoiceRecognitionHandlers,
  LiveVoiceSynthesisAdapter
} from "../../shared/liveVoice";
import type { NeuralModalityCapabilities, OmniApi } from "../../shared/types";
import {
  createBrowserLivePerceptionCapture,
  type LivePerceptionCaptureAdapter
} from "./livePerceptionController";
import { createNeuralSpeechSynthesisAdapter, type NeuralWavePlayerConstructor } from "./neuralSpeechPlayback";
import { createNativeAudioCaptureAdapter, type NativeAudioCaptureAdapter } from "./nativeAudioCapture";

interface RecognitionAlternativeLike {
  transcript: string;
  confidence?: number;
}

interface RecognitionResultLike {
  isFinal: boolean;
  length: number;
  [index: number]: RecognitionAlternativeLike;
}

interface RecognitionResultListLike {
  length: number;
  [index: number]: RecognitionResultLike;
}

interface RecognitionEventLike extends Event {
  resultIndex: number;
  results: RecognitionResultListLike;
}

interface RecognitionErrorEventLike extends Event {
  error: string;
  message?: string;
}

interface RecognitionLike {
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  lang: string;
  onspeechstart: (() => void) | null;
  onresult: ((event: RecognitionEventLike) => void) | null;
  onerror: ((event: RecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}

type RecognitionConstructor = new () => RecognitionLike;

export interface BrowserVoiceEnvironment {
  mediaDevices?: Pick<MediaDevices, "getUserMedia">;
  recognitionConstructor?: RecognitionConstructor;
  speechSynthesis?: Pick<SpeechSynthesis, "speak" | "cancel">;
  utteranceConstructor?: typeof SpeechSynthesisUtterance;
  audioConstructor?: NeuralWavePlayerConstructor;
  audioContextConstructor?: typeof AudioContext;
}

export const LIVE_VOICE_CAPTURE_REQUEST_TIMEOUT_MS = 15_000;

function runtimeEnvironment(): BrowserVoiceEnvironment {
  const scope = globalThis as typeof globalThis & {
    SpeechRecognition?: RecognitionConstructor;
    webkitSpeechRecognition?: RecognitionConstructor;
    webkitAudioContext?: typeof AudioContext;
  };
  return {
    mediaDevices: globalThis.navigator?.mediaDevices,
    recognitionConstructor:
      scope.SpeechRecognition ?? scope.webkitSpeechRecognition,
    speechSynthesis: globalThis.speechSynthesis,
    utteranceConstructor: globalThis.SpeechSynthesisUtterance,
    audioConstructor: typeof globalThis.Audio === "function"
      ? globalThis.Audio as unknown as NeuralWavePlayerConstructor : undefined,
    audioContextConstructor: globalThis.AudioContext ?? scope.webkitAudioContext
  };
}

export function inspectBrowserVoiceCapabilities(
  environment: BrowserVoiceEnvironment = runtimeEnvironment()
): LiveVoiceCapabilities {
  const capture = typeof environment.mediaDevices?.getUserMedia === "function";
  const recognition = Boolean(environment.recognitionConstructor);
  const synthesis = Boolean(
    environment.speechSynthesis && environment.utteranceConstructor
  );
  return {
    microphoneCapture: capture,
    speechRecognition: recognition,
    speechSynthesis: synthesis,
    fullDuplex: capture && recognition,
    bargeIn: synthesis,
    neuralListening: false,
    neuralVoice: false,
    recognitionDetail: recognition
      ? "Web Speech recognition is available. Transcripts come from the platform recognizer."
      : "Web Speech recognition is not exposed by this Electron/Chromium build. No transcript fallback is fabricated.",
    neuralDetail:
      "Platform STT/TTS are the defaults. Own waveform output uses this brain's audio region; speech quality requires paired training and verification."
  };
}

export function createBrowserCaptureAdapter(
  environment: BrowserVoiceEnvironment = runtimeEnvironment(),
  requestTimeoutMs = LIVE_VOICE_CAPTURE_REQUEST_TIMEOUT_MS
): LiveVoiceCaptureAdapter {
  return {
    available: typeof environment.mediaDevices?.getUserMedia === "function",
    async request() {
      if (!environment.mediaDevices) {
        throw new DOMException("Microphone capture is unavailable.", "NotSupportedError");
      }
      // Called only after the person explicitly starts live voice.
      const capture = environment.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        },
        video: false
      });
      let timeoutId: ReturnType<typeof globalThis.setTimeout> | undefined;
      const timeout = new Promise<never>((_resolve, reject) => {
        timeoutId = globalThis.setTimeout(() => {
          reject(
            new DOMException(
              "The operating system did not answer the microphone permission request.",
              "TimeoutError"
            )
          );
        }, Math.max(1, requestTimeoutMs));
      });
      let stream: MediaStream;
      try {
        stream = await Promise.race([capture, timeout]);
      } catch (error) {
        if (error instanceof DOMException && error.name === "TimeoutError") {
          // getUserMedia cannot be cancelled. If the OS answers after our bounded
          // start gate has failed, close that late stream instead of leaking it.
          void capture.then(
            (lateStream) => lateStream.getTracks().forEach((track) => track.stop()),
            () => undefined
          );
        }
        throw error;
      } finally {
        if (timeoutId !== undefined) globalThis.clearTimeout(timeoutId);
      }
      return {
        stream,
        stop: () => stream.getTracks().forEach((track) => track.stop())
      };
    }
  };
}

export function createBrowserRecognitionAdapter(
  environment: BrowserVoiceEnvironment = runtimeEnvironment(),
  language = globalThis.navigator?.language || "en-US"
): LiveVoiceRecognitionAdapter {
  const Constructor = environment.recognitionConstructor;
  const detail = Constructor
    ? "Using the runtime Web Speech recognizer. Availability and privacy behavior are platform-defined."
    : "Speech recognition is unavailable in this Electron/Chromium runtime. No transcription is simulated.";
  return {
    available: Boolean(Constructor),
    detail,
    start(handlers: LiveVoiceRecognitionHandlers) {
      if (!Constructor) throw new Error(detail);
      const recognition = new Constructor();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;
      recognition.lang = language;
      recognition.onspeechstart = handlers.onSpeechStart;
      recognition.onresult = (event) => {
        let interim = "";
        const finalized: Array<{ transcript: string; confidence?: number }> = [];
        for (let index = event.resultIndex; index < event.results.length; index += 1) {
          const result = event.results[index];
          const alternative = result?.[0];
          if (!result || !alternative) continue;
          if (result.isFinal) {
            finalized.push({
              transcript: alternative.transcript,
              confidence: alternative.confidence
            });
          } else {
            interim += alternative.transcript;
          }
        }
        if (interim) handlers.onInterim(interim);
        finalized.forEach((result) => handlers.onFinal(result));
      };
      recognition.onerror = (event) =>
        handlers.onError({ code: event.error, message: event.message });
      recognition.onend = handlers.onEnd;
      recognition.start();
      return {
        stop: () => recognition.stop(),
        abort: () => recognition.abort()
      };
    }
  };
}

export function createBrowserSynthesisAdapter(
  environment: BrowserVoiceEnvironment = runtimeEnvironment()
): LiveVoiceSynthesisAdapter {
  const available = Boolean(
    environment.speechSynthesis && environment.utteranceConstructor
  );
  return {
    available,
    cancel: () => environment.speechSynthesis?.cancel(),
    speak(text, handlers, options) {
      if (!environment.speechSynthesis || !environment.utteranceConstructor) {
        throw new Error("Speech synthesis is unavailable.");
      }
      const utterance = new environment.utteranceConstructor(text);
      utterance.rate = Math.max(0.1, Math.min(10, options.rate));
      utterance.onstart = handlers.onStart;
      utterance.onend = handlers.onEnd;
      utterance.onerror = (event) => handlers.onError(event.error);
      environment.speechSynthesis.speak(utterance);
      return { cancel: () => environment.speechSynthesis?.cancel() };
    }
  };
}

export function createOmniLiveVoiceChatAdapter(
  chat: OmniApi["chat"]
): LiveVoiceChatAdapter {
  return {
    send: (brainId, transcript, turnId) =>
      chat.send(brainId, transcript, turnId),
    ...(typeof chat.sendAudio === "function" ? { sendAudio: (brainId: string, input: import("../../shared/nativeAudioChat").NativeAudioChatRequest, turnId: string) => chat.sendAudio(brainId, input, turnId) } : {}),
    cancel: (brainId, turnId) => chat.cancel(brainId, turnId),
    subscribe: (listener) => chat.onStream(listener)
  };
}

function runtimeOmniModality(): OmniApi["modality"] | undefined {
  const scope = globalThis as typeof globalThis & {
    window?: { omni?: OmniApi };
  };
  return scope.window?.omni?.modality;
}

function runtimeOmniChat(): OmniApi["chat"] | undefined {
  return (globalThis as typeof globalThis & { window?: { omni?: OmniApi } }).window?.omni?.chat;
}

/**
 * Feeds raw microphone audio into this brain's trained sensory substrate. It
 * never invokes a platform recognizer. Actual audio conditions the same brain's
 * ordinary chat route. Output quality is separate from connectivity.
 */
export function createOmniNeuralAudioAdapter(
  modality: OmniApi["modality"] | undefined = runtimeOmniModality(),
  _capture: LivePerceptionCaptureAdapter = createBrowserLivePerceptionCapture(),
  options: { brainId?: string; audioConstructor?: NeuralWavePlayerConstructor; chat?: OmniApi["chat"]; nativeCapture?: NativeAudioCaptureAdapter } = {}
): LiveVoiceNeuralAdapter {
  const measured = new Map<string, NeuralModalityCapabilities>();
  let selectedBrainId = options.brainId;
  const chat = options.chat ?? runtimeOmniChat();
  const rawCapture = options.nativeCapture ?? createNativeAudioCaptureAdapter(createBrowserCaptureAdapter());
  let nativeInputAvailable = false;
  const synthesis = createNeuralSpeechSynthesisAdapter(() => selectedBrainId, modality,
    options.audioConstructor ?? runtimeEnvironment().audioConstructor);

  const inspect = async (
    brainId: string
  ): Promise<NeuralModalityCapabilities | undefined> => {
    if (!modality) return undefined;
    const capabilities = await modality.capabilities(brainId);
    measured.set(brainId, capabilities);
    return capabilities;
  };

  return {
    async inspect(brainId) {
      if (options.brainId && options.brainId !== brainId) throw new Error("Neural voice is bound to another mind.");
      selectedBrainId = brainId;
      if (!modality) {
        return {
          listening: false,
          voice: false,
          detail:
            "The trusted neural-media preload API is unavailable. Platform STT/TTS remain separate."
        };
      }
      const capabilities = await inspect(brainId);
      const native = chat?.audioInputCapabilities ? await chat.audioInputCapabilities(brainId) : undefined;
      nativeInputAvailable = native?.brainId === brainId && native.loaded === true && native.available === true &&
        native.transcriptRequired === false && rawCapture.available && typeof chat?.sendAudio === "function";
      const listening = nativeInputAvailable;
      const voice = capabilities?.audioRegionAvailable === true && synthesis.available;
      const quality = capabilities?.speechQuality ?? "needs-speech-training";
      return {
        listening,
        nativeAudioChatInput: nativeInputAvailable,
        nativeAudioChatInputDetail: native?.detail ?? "Direct raw-audio chat input is unavailable; no platform transcript was substituted.",
        voice,
        voiceQuality: quality,
        detail: voice
          ? quality === "needs-speech-training"
            ? "Own waveform output uses this brain's audio region and needs paired speech training. It may not match the words yet. STT/TTS remain defaults."
            : "Own waveform output uses this brain's audio region. Speech quality is not verified; STT/TTS remain defaults."
          : "Same-brain waveform output or local playback is unavailable. Platform STT/TTS remain separate."
      };
    },
    async startListening(brainId, handlers, signal) {
      if (!modality) {
        throw new Error("The trusted neural-media preload API is unavailable.");
      }
      if (!nativeInputAvailable || !handlers || !chat?.sendAudio) throw new Error("Direct native audio conversation is unavailable; STT was not substituted.");
      const capabilities = measured.get(brainId) ?? (await inspect(brainId));
      if (!capabilities?.audioRegionAvailable) throw new Error("This brain's own audio region is disabled.");
      return rawCapture.start(async (sampleRate) => {
        const lease = await modality.startObservation({ brainId, modalities: ["audio"], retention: "working",
          permission: { granted: true, scope: "session", source: "microphone", grantedAt: new Date().toISOString() },
          capture: { mode: "auto", audioSampleRate: sampleRate, benchmarkClass: "balanced" } });
        return { sessionId: lease.id, maxBytes: lease.maxPacketBytes, stop: () => modality.stopObservation(lease.id) };
      }, handlers, signal);
    },
    ...(synthesis.available ? { synthesis } : {})
  };
}

export function createBrowserLiveVoiceAdapters(
  environment: BrowserVoiceEnvironment = runtimeEnvironment(),
  brainId?: string
): {
  capture: LiveVoiceCaptureAdapter;
  recognition: LiveVoiceRecognitionAdapter;
  synthesis: LiveVoiceSynthesisAdapter;
  neural: LiveVoiceNeuralAdapter;
} {
  const voiceCapture = createBrowserCaptureAdapter(environment);
  return {
    capture: voiceCapture,
    recognition: createBrowserRecognitionAdapter(environment),
    synthesis: createBrowserSynthesisAdapter(environment),
    neural: createOmniNeuralAudioAdapter(runtimeOmniModality(), createBrowserLivePerceptionCapture(), {
      brainId, audioConstructor: environment.audioConstructor,
      nativeCapture: createNativeAudioCaptureAdapter(voiceCapture, environment.audioContextConstructor)
    })
  };
}
