import { describe, expect, it, vi } from "vitest";
import type {
  ChatResult,
  LiveObservationSession,
  NeuralModalityCapabilities,
  OmniApi
} from "../src/shared/types";
import {
  createBrowserCaptureAdapter,
  createBrowserRecognitionAdapter,
  createBrowserSynthesisAdapter,
  createOmniLiveVoiceChatAdapter,
  createOmniNeuralAudioAdapter,
  inspectBrowserVoiceCapabilities,
  type BrowserVoiceEnvironment
} from "../src/renderer/src/browserLiveVoice";
import type {
  LivePerceptionCaptureAdapter,
  LivePerceptionCaptureHandle
} from "../src/renderer/src/livePerceptionController";

function neuralCapabilities(
  patch: Partial<NeuralModalityCapabilities> = {}
): NeuralModalityCapabilities {
  return {
    brainId: "brain-neural",
    hardwareTier: "personal",
    imagePerception: false,
    audioPerception: true,
    videoPerception: false,
    imageGeneration: false,
    audioGeneration: true,
    videoGeneration: false,
    neuralSpeechRecognition: false,
    neuralSpeechSynthesis: false,
    synchronizedVideoAudioGeneration: false,
    sameBrainSubstrate: true,
    hiddenBehavioralPrompt: false,
    detail: "same brain fixture",
    ...patch
  };
}

function observationSession(
  state: LiveObservationSession["state"] = "active"
): LiveObservationSession {
  const now = "2026-08-23T00:00:00.000Z";
  return {
    id: "neural-listening-session",
    brainId: "brain-neural",
    state,
    modalities: ["audio"],
    permission: {
      source: "microphone",
      granted: true,
      scope: "session",
      grantedAt: now
    },
    retention: "neural",
    maxInFlight: 1,
    maxPacketBytes: 1024 * 1024,
    inFlight: 0,
    packetsReceived: 0,
    packetsAccepted: 0,
    packetsDroppedBackpressure: 0,
    bytesAccepted: 0,
    lastSequence: -1,
    createdAt: now,
    updatedAt: now,
    capabilities: {
      imageNeural: false,
      audioNeural: true,
      videoNeural: false
    },
    capture: {
      mode: "auto",
      audioSampleRate: 48_000,
      benchmarkClass: "balanced",
      revision: 0
    }
  };
}

describe("browser live voice adapters", () => {
  it("exposes an explicit unavailable capability instead of fake recognition", () => {
    expect(inspectBrowserVoiceCapabilities({})).toEqual({
      microphoneCapture: false,
      speechRecognition: false,
      speechSynthesis: false,
      fullDuplex: false,
      bargeIn: false,
      neuralListening: false,
      neuralVoice: false,
      recognitionDetail:
        "Web Speech recognition is not exposed by this Electron/Chromium build. No transcript fallback is fabricated.",
      neuralDetail:
        "Platform Web Speech/TTS is not neural audio. Direct trained-modality input/output requires a compatible neural audio pack."
    });
    expect(createBrowserRecognitionAdapter({}).available).toBe(false);
  });

  it("opens audio-only capture on request and stops every returned track", async () => {
    const stopOne = vi.fn();
    const stopTwo = vi.fn();
    const getUserMedia = vi.fn(async () => ({
      getTracks: () => [{ stop: stopOne }, { stop: stopTwo }]
    })) as unknown as MediaDevices["getUserMedia"];
    const adapter = createBrowserCaptureAdapter({
      mediaDevices: { getUserMedia }
    });

    expect(getUserMedia).not.toHaveBeenCalled();
    const session = await adapter.request();
    expect(getUserMedia).toHaveBeenCalledWith({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      },
      video: false
    });
    session.stop();
    expect(stopOne).toHaveBeenCalledOnce();
    expect(stopTwo).toHaveBeenCalledOnce();
  });

  it("bounds an unanswered OS permission request and closes a late stream", async () => {
    vi.useFakeTimers();
    let resolvePending!: (stream: MediaStream) => void;
    const pending = new Promise<MediaStream>((resolve) => {
      resolvePending = resolve;
    });
    const stop = vi.fn();
    const getUserMedia = vi.fn(() => pending) as unknown as MediaDevices["getUserMedia"];
    const adapter = createBrowserCaptureAdapter(
      { mediaDevices: { getUserMedia } },
      25
    );

    const request = adapter.request();
    const rejection = expect(request).rejects.toMatchObject({ name: "TimeoutError" });
    await vi.advanceTimersByTimeAsync(25);
    await rejection;

    resolvePending({
      getTracks: () => [{ stop }]
    } as unknown as MediaStream);
    await Promise.resolve();
    expect(stop).toHaveBeenCalledOnce();
    vi.useRealTimers();
  });

  it("configures continuous interim recognition and forwards genuine result chunks", () => {
    let instance: {
      continuous: boolean;
      interimResults: boolean;
      maxAlternatives: number;
      lang: string;
      onspeechstart: (() => void) | null;
      onresult: ((event: never) => void) | null;
      onerror: ((event: never) => void) | null;
      onend: (() => void) | null;
      start: ReturnType<typeof vi.fn>;
      stop: ReturnType<typeof vi.fn>;
      abort: ReturnType<typeof vi.fn>;
    } | undefined;
    class Recognition {
      continuous = false;
      interimResults = false;
      maxAlternatives = 0;
      lang = "";
      onspeechstart: (() => void) | null = null;
      onresult: ((event: never) => void) | null = null;
      onerror: ((event: never) => void) | null = null;
      onend: (() => void) | null = null;
      start = vi.fn();
      stop = vi.fn();
      abort = vi.fn();
      constructor() {
        instance = this;
      }
    }
    const environment = {
      recognitionConstructor: Recognition
    } as unknown as BrowserVoiceEnvironment;
    const handlers = {
      onSpeechStart: vi.fn(),
      onInterim: vi.fn(),
      onFinal: vi.fn(),
      onError: vi.fn(),
      onEnd: vi.fn()
    };
    const session = createBrowserRecognitionAdapter(environment, "en-GB").start(
      handlers
    );

    expect(instance).toMatchObject({
      continuous: true,
      interimResults: true,
      maxAlternatives: 1,
      lang: "en-GB"
    });
    expect(instance?.start).toHaveBeenCalledOnce();
    instance?.onspeechstart?.();
    instance?.onresult?.({
      resultIndex: 0,
      results: {
        length: 2,
        0: { isFinal: false, length: 1, 0: { transcript: "still speaking " } },
        1: {
          isFinal: true,
          length: 1,
          0: { transcript: "final chunk", confidence: 0.88 }
        }
      }
    } as never);
    expect(handlers.onSpeechStart).toHaveBeenCalledOnce();
    expect(handlers.onInterim).toHaveBeenCalledWith("still speaking ");
    expect(handlers.onFinal).toHaveBeenCalledWith({
      transcript: "final chunk",
      confidence: 0.88
    });

    void Promise.resolve(session).then((active) => {
      active.stop();
      active.abort();
      expect(instance?.stop).toHaveBeenCalledOnce();
      expect(instance?.abort).toHaveBeenCalledOnce();
    });
  });

  it("passes the exact transcript and correlation to the existing typed chat API", async () => {
    const chat = {
      send: vi.fn(async () => ({ brainMessage: { content: "reply" } }) as ChatResult),
      cancel: vi.fn(async () => 1)
    } as unknown as OmniApi["chat"];
    const adapter = createOmniLiveVoiceChatAdapter(chat);

    await adapter.send("brain", "recognized exactly", "voice-one");
    await adapter.cancel("brain", "voice-one");
    expect(chat.send).toHaveBeenCalledWith(
      "brain",
      "recognized exactly",
      "voice-one"
    );
    expect(chat.cancel).toHaveBeenCalledWith("brain", "voice-one");
  });

  it("opens a persistent same-brain neural microphone stream without relabeling it as ASR or TTS", async () => {
    const handleStop = vi.fn(async () => undefined);
    const handle: LivePerceptionCaptureHandle = {
      getCapture: () => ({
        mode: "auto",
        audioSampleRate: 48_000,
        benchmarkClass: "balanced",
        revision: 0
      }),
      setPacketByteLimit: vi.fn(),
      start: vi.fn(),
      reconfigure: vi.fn(async () => {
        throw new Error("Audio-only capture cannot be visually reconfigured.");
      }),
      snapshotBurst: vi.fn(async () => {
        throw new Error("Audio-only capture has no snapshots.");
      }),
      stop: handleStop
    };
    const capture: LivePerceptionCaptureAdapter = {
      capabilities: { camera: false, microphone: true, screen: false },
      open: vi.fn(async () => handle)
    };
    const startObservation = vi.fn(async () => observationSession());
    const cancelObservation = vi.fn(async () => observationSession("cancelled"));
    const modality = {
      capabilities: vi.fn(async () => neuralCapabilities()),
      startObservation,
      pushObservation: vi.fn(),
      stopObservation: vi.fn(async () => observationSession("stopped")),
      cancelObservation,
      requestObservationControl: vi.fn(),
      resolveObservationControl: vi.fn(),
      onObservation: vi.fn(() => () => undefined)
    } as unknown as OmniApi["modality"];
    const adapter = createOmniNeuralAudioAdapter(modality, capture);

    await expect(adapter.inspect("brain-neural")).resolves.toMatchObject({
      listening: true,
      voice: false
    });
    const session = await adapter.startListening("brain-neural");
    expect(startObservation).toHaveBeenCalledWith(
      expect.objectContaining({
        brainId: "brain-neural",
        modalities: ["audio"],
        retention: "neural",
        permission: expect.objectContaining({ source: "microphone" })
      })
    );
    expect(capture.open).toHaveBeenCalledWith(
      "microphone",
      expect.objectContaining({ realtime: true })
    );
    expect(handle.start).toHaveBeenCalledOnce();

    session.stop();
    await vi.waitFor(() =>
      expect(cancelObservation).toHaveBeenCalledWith("neural-listening-session")
    );
    expect(handleStop).toHaveBeenCalledOnce();
  });

  it("does not advertise generic neural audio generation as intelligible voice", async () => {
    const modality = {
      capabilities: vi.fn(async () =>
        neuralCapabilities({
          audioGeneration: true,
          neuralSpeechSynthesis: false
        })
      )
    } as unknown as OmniApi["modality"];
    const adapter = createOmniNeuralAudioAdapter(modality, {
      capabilities: { camera: false, microphone: true, screen: false },
      open: vi.fn()
    });

    await expect(adapter.inspect("brain-neural")).resolves.toMatchObject({
      listening: true,
      voice: false
    });
    expect(adapter.synthesis).toBeUndefined();
  });

  it.each([
    ["slow", 0.8],
    ["normal", 1],
    ["fast", 1.35]
  ])("applies the %s synthesis rate without changing spoken text", (_pace, rate) => {
    let utterance: { text: string; rate: number } | undefined;
    class FakeUtterance {
      rate = 1;
      onstart: (() => void) | null = null;
      onend: (() => void) | null = null;
      onerror: ((event: { error: string }) => void) | null = null;
      constructor(readonly text: string) {
        utterance = this;
      }
    }
    const speak = vi.fn();
    const adapter = createBrowserSynthesisAdapter({
      speechSynthesis: { speak, cancel: vi.fn() } as unknown as SpeechSynthesis,
      utteranceConstructor:
        FakeUtterance as unknown as typeof SpeechSynthesisUtterance
    });
    adapter.speak(
      "Exact brain reply.",
      { onStart: vi.fn(), onEnd: vi.fn(), onError: vi.fn() },
      { rate }
    );

    expect(speak).toHaveBeenCalledOnce();
    expect(utterance).toMatchObject({
      text: "Exact brain reply.",
      rate
    });
  });
});
