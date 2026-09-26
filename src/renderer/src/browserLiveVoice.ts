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
  LivePerceptionController,
  planLivePerception,
  type LivePerceptionCaptureAdapter
} from "./livePerceptionController";

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
}

export const LIVE_VOICE_CAPTURE_REQUEST_TIMEOUT_MS = 15_000;

function runtimeEnvironment(): BrowserVoiceEnvironment {
  const scope = globalThis as typeof globalThis & {
    SpeechRecognition?: RecognitionConstructor;
    webkitSpeechRecognition?: RecognitionConstructor;
  };
  return {
    mediaDevices: globalThis.navigator?.mediaDevices,
    recognitionConstructor:
      scope.SpeechRecognition ?? scope.webkitSpeechRecognition,
    speechSynthesis: globalThis.speechSynthesis,
    utteranceConstructor: globalThis.SpeechSynthesisUtterance
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
      "Platform Web Speech/TTS is not neural audio. Direct trained-modality input/output requires a compatible neural audio pack."
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

/**
 * Feeds raw microphone audio into this brain's trained sensory substrate. It
 * deliberately does not transcribe or synthesize speech: Web Speech remains
 * the language-boundary transcript until a verified speech pack exists.
 */
export function createOmniNeuralAudioAdapter(
  modality: OmniApi["modality"] | undefined = runtimeOmniModality(),
  capture: LivePerceptionCaptureAdapter = createBrowserLivePerceptionCapture()
): LiveVoiceNeuralAdapter {
  const measured = new Map<string, NeuralModalityCapabilities>();

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
      if (!modality) {
        return {
          listening: false,
          voice: false,
          detail:
            "The trusted neural-media preload API is unavailable. Platform STT/TTS remain separate."
        };
      }
      if (!capture.capabilities.microphone) {
        return {
          listening: false,
          voice: false,
          detail:
            "This runtime cannot encode direct microphone packets with MediaRecorder. Platform STT/TTS remain separate."
        };
      }
      const capabilities = await inspect(brainId);
      if (!capabilities?.audioPerception) {
        return {
          listening: false,
          voice: false,
          detail:
            "This brain has no verified trained neural audio-input path. Platform STT/TTS remain separate."
        };
      }
      return {
        listening: true,
        // The built-in audio decoder creates general sound, not intelligible
        // speech. Do not turn a generic audio capability into fake TTS.
        // No production intelligible-speech renderer is installed in this
        // adapter yet, even if a future worker can report such a pack.
        voice: false,
        detail:
          "Microphone waveforms enter this brain's persistent assemblies and STDP. Platform Web Speech supplies only the literal transcript; generic neural audio is not TTS."
      };
    },
    async startListening(brainId) {
      if (!modality) {
        throw new Error("The trusted neural-media preload API is unavailable.");
      }
      const capabilities = measured.get(brainId) ?? (await inspect(brainId));
      if (!capabilities?.audioPerception) {
        throw new Error("This brain has no verified trained neural audio-input path.");
      }
      if (!capture.capabilities.microphone) {
        throw new Error("Direct neural microphone packet encoding is unavailable.");
      }
      const controller = new LivePerceptionController({
        brainId,
        modality,
        capture
      });
      await controller.start(
        "microphone",
        planLivePerception({ hardwareTier: capabilities.hardwareTier }),
        "neural"
      );
      const state = controller.getState();
      if (!state.enabled) {
        const message = state.error ?? "Direct neural listening could not start.";
        await controller.dispose();
        throw new Error(message);
      }
      return {
        stop: () => {
          void controller.dispose();
        }
      };
    }
  };
}

export function createBrowserLiveVoiceAdapters(
  environment: BrowserVoiceEnvironment = runtimeEnvironment()
): {
  capture: LiveVoiceCaptureAdapter;
  recognition: LiveVoiceRecognitionAdapter;
  synthesis: LiveVoiceSynthesisAdapter;
  neural: LiveVoiceNeuralAdapter;
} {
  return {
    capture: createBrowserCaptureAdapter(environment),
    recognition: createBrowserRecognitionAdapter(environment),
    synthesis: createBrowserSynthesisAdapter(environment),
    neural: createOmniNeuralAudioAdapter()
  };
}
