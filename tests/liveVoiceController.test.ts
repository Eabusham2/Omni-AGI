import { describe, expect, it, vi } from "vitest";
import type {
  LiveVoiceCaptureAdapter,
  LiveVoiceChatAdapter,
  LiveVoiceNeuralAdapter,
  LiveVoiceRecognitionAdapter,
  LiveVoiceRecognitionHandlers,
  LiveVoicePreferences,
  LiveVoiceSynthesisAdapter,
  LiveVoiceSynthesisHandlers
} from "../src/shared/liveVoice";
import type { ChatResult, ChatStreamEvent } from "../src/shared/types";
import { LiveVoiceController } from "../src/renderer/src/liveVoiceController";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function result(text: string): ChatResult {
  return {
    brainMessage: { content: text }
  } as ChatResult;
}

function preferences(
  patch: Partial<Omit<LiveVoicePreferences, "schemaVersion">> = {}
): LiveVoicePreferences {
  return {
    schemaVersion: 1,
    deliveryMode: "live",
    pace: "normal",
    neuralListening: false,
    neuralVoice: false,
    ...patch
  };
}

function token(turnId: string, sequence: number, delta: string): ChatStreamEvent {
  return {
    id: `stream-${sequence}`,
    brainId: "brain-one",
    turnId,
    sequence,
    createdAt: new Date(sequence * 1_000).toISOString(),
    type: "chat-token",
    delta
  };
}

interface Harness {
  controller: LiveVoiceController;
  capture: LiveVoiceCaptureAdapter;
  recognition: LiveVoiceRecognitionAdapter;
  synthesis: LiveVoiceSynthesisAdapter;
  chat: LiveVoiceChatAdapter;
  captureStop: ReturnType<typeof vi.fn>;
  recognitionStop: ReturnType<typeof vi.fn>;
  recognitionAbort: ReturnType<typeof vi.fn>;
  synthesisCancel: ReturnType<typeof vi.fn>;
  synthesisSessionCancel: ReturnType<typeof vi.fn>;
  handlers(): LiveVoiceRecognitionHandlers;
  speechHandlers(): LiveVoiceSynthesisHandlers;
  emitStream(event: ChatStreamEvent): void;
}

function harness(overrides?: {
  recognitionAvailable?: boolean;
  synthesisAvailable?: boolean;
  captureAvailable?: boolean;
  captureRequest?: LiveVoiceCaptureAdapter["request"];
  send?: LiveVoiceChatAdapter["send"];
  scheduleRecognitionRestart?: (callback: () => void) => void;
  onAcceptedReply?: ConstructorParameters<typeof LiveVoiceController>[0]["onAcceptedReply"];
  preferences?: LiveVoicePreferences;
  maxLiveSpeechQueue?: number;
  liveSpeechChunkCharacters?: number;
  neural?: LiveVoiceNeuralAdapter;
}): Harness {
  let recognitionHandlers: LiveVoiceRecognitionHandlers | undefined;
  let synthesisHandlers: LiveVoiceSynthesisHandlers | undefined;
  let streamListener: ((event: ChatStreamEvent) => void) | undefined;
  const captureStop = vi.fn();
  const recognitionStop = vi.fn();
  const recognitionAbort = vi.fn();
  const synthesisCancel = vi.fn();
  const synthesisSessionCancel = vi.fn();
  const capture: LiveVoiceCaptureAdapter = {
    available: overrides?.captureAvailable ?? true,
    request:
      overrides?.captureRequest ??
      vi.fn(async () => ({ stop: captureStop }))
  };
  const recognition: LiveVoiceRecognitionAdapter = {
    available: overrides?.recognitionAvailable ?? true,
    detail:
      overrides?.recognitionAvailable === false
        ? "Recognition is genuinely unavailable; no transcript is fabricated."
        : "test recognizer",
    start: vi.fn((handlers) => {
      recognitionHandlers = handlers;
      return {
        stop: recognitionStop,
        abort: recognitionAbort
      };
    })
  };
  const synthesis: LiveVoiceSynthesisAdapter = {
    available: overrides?.synthesisAvailable ?? true,
    cancel: synthesisCancel,
    speak: vi.fn((_text, handlers) => {
      synthesisHandlers = handlers;
      handlers.onStart();
      return { cancel: synthesisSessionCancel };
    })
  };
  const chat: LiveVoiceChatAdapter = {
    send:
      overrides?.send ??
      vi.fn(async () => result("A spoken brain reply.")),
    cancel: vi.fn(async () => 1),
    subscribe: vi.fn((listener) => {
      streamListener = listener;
      return () => {
        streamListener = undefined;
      };
    })
  };
  let nextId = 0;
  let timestamp = 0;
  const controller = new LiveVoiceController({
    brainId: "brain-one",
    capture,
    recognition,
    synthesis,
    neural: overrides?.neural,
    chat,
    preferences: overrides?.preferences,
    maxLiveSpeechQueue: overrides?.maxLiveSpeechQueue,
    liveSpeechChunkCharacters: overrides?.liveSpeechChunkCharacters,
    createId: () => `utterance-${++nextId}`,
    now: () => new Date(++timestamp * 1_000),
    scheduleRecognitionRestart: overrides?.scheduleRecognitionRestart,
    onAcceptedReply: overrides?.onAcceptedReply
  });
  return {
    controller,
    capture,
    recognition,
    synthesis,
    chat,
    captureStop,
    recognitionStop,
    recognitionAbort,
    synthesisCancel,
    synthesisSessionCancel,
    handlers: () => {
      if (!recognitionHandlers) throw new Error("recognition has not started");
      return recognitionHandlers;
    },
    speechHandlers: () => {
      if (!synthesisHandlers) throw new Error("speech has not started");
      return synthesisHandlers;
    },
    emitStream: (event) => streamListener?.(event)
  };
}

describe("LiveVoiceController", () => {
  it("requests microphone access only after an explicit start and enters continuous listening", async () => {
    const voice = harness();
    const states: string[] = [];
    voice.controller.subscribe((state) => states.push(`${state.phase}:${state.capture}`));

    expect(voice.capture.request).not.toHaveBeenCalled();
    expect(voice.recognition.start).not.toHaveBeenCalled();
    await voice.controller.start();

    expect(voice.capture.request).toHaveBeenCalledOnce();
    expect(voice.recognition.start).toHaveBeenCalledOnce();
    expect(states).toContain("starting:requesting");
    expect(voice.controller.getState()).toMatchObject({
      enabled: true,
      phase: "listening",
      capture: "active",
      recognition: "listening",
      hiddenBehavioralPrompt: false
    });
  });

  it("reports honest recognition unavailability without requesting the microphone", async () => {
    const voice = harness({ recognitionAvailable: false });
    await voice.controller.start();

    expect(voice.capture.request).not.toHaveBeenCalled();
    expect(voice.chat.send).not.toHaveBeenCalled();
    expect(voice.controller.getState()).toMatchObject({
      enabled: false,
      phase: "unavailable",
      recognition: "unavailable",
      error: {
        code: "recognition-unavailable",
        recoverable: false
      }
    });
    expect(voice.controller.getState().error?.message).toContain("no transcript is fabricated");
  });

  it("maps microphone denial to a recoverable permission state", async () => {
    const denied = new DOMException("Denied by the person", "NotAllowedError");
    const voice = harness({ captureRequest: vi.fn(async () => Promise.reject(denied)) });
    await voice.controller.start();

    expect(voice.recognition.start).not.toHaveBeenCalled();
    expect(voice.controller.getState()).toMatchObject({
      enabled: false,
      phase: "error",
      capture: "error",
      error: { code: "permission-denied", recoverable: true }
    });
  });

  it("maps an unanswered microphone prompt to actionable recovery", async () => {
    const timeout = new DOMException("OS did not answer", "TimeoutError");
    const voice = harness({ captureRequest: vi.fn(async () => Promise.reject(timeout)) });
    await voice.controller.start();

    expect(voice.controller.getState()).toMatchObject({
      enabled: false,
      phase: "error",
      capture: "error",
      error: {
        code: "permission-timeout",
        recoverable: true
      }
    });
    expect(voice.controller.getState().error?.message).toContain("allow Electron");
  });

  it("sends each finalized transcript verbatim with a unique voice turn and speaks only its accepted reply", async () => {
    const accepted = vi.fn();
    const voice = harness({ onAcceptedReply: accepted });
    await voice.controller.start();

    voice.handlers().onSpeechStart();
    voice.handlers().onInterim("   exact recognized   ");
    expect(voice.controller.getState().interimTranscript).toBe("exact recognized");
    voice.handlers().onFinal({
      transcript: "  Keep these exact recognized words.  ",
      confidence: 2
    });

    await vi.waitFor(() => expect(voice.synthesis.speak).toHaveBeenCalledOnce());
    expect(voice.chat.send).toHaveBeenCalledWith(
      "brain-one",
      "Keep these exact recognized words.",
      "voice-utterance-1"
    );
    expect(voice.synthesis.speak).toHaveBeenCalledWith(
      "A spoken brain reply.",
      expect.any(Object),
      { rate: 1 }
    );
    expect(accepted).toHaveBeenCalledOnce();
    expect(voice.controller.getState()).toMatchObject({
      phase: "speaking",
      activeUtterance: {
        id: "utterance-1",
        turnId: "voice-utterance-1",
        sequence: 1,
        confidence: 1
      },
      lastReply: {
        utteranceId: "utterance-1",
        text: "A spoken brain reply."
      },
      hiddenBehavioralPrompt: false
    });

    voice.speechHandlers().onEnd();
    expect(voice.controller.getState()).toMatchObject({
      phase: "listening",
      activeUtterance: undefined,
      lastUtterance: { id: "utterance-1" }
    });
  });

  it("does not let a same-batch interim fragment demote a finalized Ponder turn", async () => {
    const pending = deferred<ChatResult>();
    const voice = harness({ send: vi.fn(() => pending.promise) });
    await voice.controller.start();
    voice.handlers().onFinal({ transcript: "finalized words" });
    voice.handlers().onInterim("stale interim from the same recognition event");
    await vi.waitFor(() => expect(voice.chat.send).toHaveBeenCalledOnce());

    expect(voice.controller.getState()).toMatchObject({
      phase: "pondering",
      interimTranscript: "",
      activeUtterance: { transcript: "finalized words" }
    });
  });

  it("barge-in synchronously silences TTS and cancels the correlated turn", async () => {
    const voice = harness();
    await voice.controller.start();
    voice.handlers().onFinal({ transcript: "first utterance" });
    await vi.waitFor(() => expect(voice.synthesis.speak).toHaveBeenCalledOnce());

    voice.handlers().onSpeechStart();

    expect(voice.synthesisSessionCancel).toHaveBeenCalledOnce();
    expect(voice.synthesisCancel).toHaveBeenCalled();
    expect(voice.chat.cancel).toHaveBeenCalledWith(
      "brain-one",
      "voice-utterance-1"
    );
    expect(voice.controller.getState()).toMatchObject({
      enabled: true,
      phase: "listening",
      activeUtterance: undefined,
      synthesis: "idle"
    });
  });

  it("discards a late reply after a newer utterance fork has started", async () => {
    const first = deferred<ChatResult>();
    const second = deferred<ChatResult>();
    const send = vi
      .fn<LiveVoiceChatAdapter["send"]>()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const accepted = vi.fn();
    const voice = harness({ send, onAcceptedReply: accepted });
    await voice.controller.start();

    voice.handlers().onFinal({ transcript: "old request" });
    await vi.waitFor(() => expect(send).toHaveBeenCalledTimes(1));
    voice.handlers().onSpeechStart();
    voice.handlers().onFinal({ transcript: "new request" });
    await vi.waitFor(() => expect(send).toHaveBeenCalledTimes(2));

    first.resolve(result("obsolete reply"));
    await Promise.resolve();
    expect(accepted).not.toHaveBeenCalled();
    expect(voice.synthesis.speak).not.toHaveBeenCalled();

    second.resolve(result("current reply"));
    await vi.waitFor(() => expect(accepted).toHaveBeenCalledOnce());
    expect(voice.synthesis.speak).toHaveBeenCalledWith(
      "current reply",
      expect.any(Object),
      { rate: 1 }
    );
    expect(voice.controller.getState().lastReply).toMatchObject({
      utteranceId: "utterance-2",
      turnId: "voice-utterance-2",
      text: "current reply"
    });
  });

  it("stops capture, recognition, synthesis, and an in-flight chat turn", async () => {
    const pending = deferred<ChatResult>();
    const voice = harness({ send: vi.fn(() => pending.promise) });
    await voice.controller.start();
    voice.handlers().onFinal({ transcript: "cancel this turn" });
    await vi.waitFor(() => expect(voice.chat.send).toHaveBeenCalledOnce());

    await voice.controller.stop();

    expect(voice.captureStop).toHaveBeenCalledOnce();
    expect(voice.recognitionAbort).toHaveBeenCalledOnce();
    expect(voice.synthesisCancel).toHaveBeenCalled();
    expect(voice.chat.cancel).toHaveBeenCalledWith(
      "brain-one",
      "voice-utterance-1"
    );
    expect(voice.controller.getState()).toMatchObject({
      enabled: false,
      phase: "idle",
      capture: "idle",
      recognition: "idle",
      activeUtterance: undefined
    });

    pending.resolve(result("too late"));
    await Promise.resolve();
    expect(voice.synthesis.speak).not.toHaveBeenCalled();
  });

  it("restarts recognition after an engine end but invalidates queued restarts on stop", async () => {
    const restarts: Array<() => void> = [];
    const voice = harness({
      scheduleRecognitionRestart: (callback) => restarts.push(callback)
    });
    await voice.controller.start();
    voice.handlers().onEnd();
    expect(restarts).toHaveLength(1);
    restarts.shift()?.();
    await vi.waitFor(() => expect(voice.recognition.start).toHaveBeenCalledTimes(2));

    voice.handlers().onEnd();
    expect(restarts).toHaveLength(1);
    await voice.controller.stop();
    restarts.shift()?.();
    await Promise.resolve();
    expect(voice.recognition.start).toHaveBeenCalledTimes(2);
  });

  it("continues after no-speech but tears down devices after a fatal recognition error", async () => {
    const voice = harness();
    await voice.controller.start();
    voice.handlers().onError({ code: "no-speech" });
    await vi.waitFor(() =>
      expect(voice.controller.getState().error?.message).toContain("continues")
    );
    expect(voice.controller.getState()).toMatchObject({
      enabled: true,
      recognition: "listening"
    });

    voice.handlers().onError({ code: "audio-capture" });
    await vi.waitFor(() => expect(voice.controller.getState().phase).toBe("error"));
    expect(voice.captureStop).toHaveBeenCalledOnce();
    expect(voice.recognitionAbort).toHaveBeenCalledOnce();
    expect(voice.controller.getState()).toMatchObject({
      enabled: false,
      error: { code: "no-input-device" }
    });
  });

  it("accepts text replies without pretending TTS exists", async () => {
    const accepted = vi.fn();
    const voice = harness({ synthesisAvailable: false, onAcceptedReply: accepted });
    await voice.controller.start();
    voice.handlers().onFinal({ transcript: "text-only voice reply" });
    await vi.waitFor(() => expect(accepted).toHaveBeenCalledOnce());

    expect(voice.synthesis.speak).not.toHaveBeenCalled();
    expect(voice.controller.getState()).toMatchObject({
      enabled: true,
      phase: "listening",
      synthesis: "unavailable",
      lastReply: { text: "A spoken brain reply." }
    });
  });

  it("speaks live chunks before generation completes and applies the selected pace", async () => {
    const pending = deferred<ChatResult>();
    const accepted = vi.fn();
    const voice = harness({
      send: vi.fn(() => pending.promise),
      preferences: preferences({ deliveryMode: "live", pace: "fast" }),
      onAcceptedReply: accepted
    });
    await voice.controller.start();
    voice.handlers().onFinal({ transcript: "stream this reply" });
    await vi.waitFor(() => expect(voice.chat.send).toHaveBeenCalledOnce());

    voice.emitStream(token("voice-utterance-1", 0, "First sentence arrives. "));
    await vi.waitFor(() => expect(voice.synthesis.speak).toHaveBeenCalledOnce());
    expect(voice.synthesis.speak).toHaveBeenCalledWith(
      "First sentence arrives.",
      expect.any(Object),
      { rate: 1.35 }
    );
    expect(accepted).not.toHaveBeenCalled();
    expect(voice.controller.getState().phase).toBe("speaking");

    pending.resolve(result("First sentence arrives."));
    await vi.waitFor(() => expect(accepted).toHaveBeenCalledOnce());
    voice.speechHandlers().onEnd();
    expect(voice.controller.getState()).toMatchObject({
      phase: "listening",
      activeUtterance: undefined
    });
  });

  it("bounds the native speech backlog without dropping generated speech", async () => {
    const pending = deferred<ChatResult>();
    const voice = harness({
      send: vi.fn(() => pending.promise),
      preferences: preferences({ deliveryMode: "live" }),
      maxLiveSpeechQueue: 2,
      liveSpeechChunkCharacters: 48
    });
    await voice.controller.start();
    voice.handlers().onFinal({ transcript: "generate much faster than speech" });
    await vi.waitFor(() => expect(voice.chat.send).toHaveBeenCalledOnce());

    let fullReply = "";
    for (let sequence = 0; sequence < 9; sequence += 1) {
      const delta =
        `Sentence ${sequence} contains enough generated words for incremental speech. `;
      fullReply += delta;
      voice.emitStream(
        token(
          "voice-utterance-1",
          sequence,
          delta
        )
      );
    }

    expect(voice.synthesis.speak).toHaveBeenCalledTimes(1);
    expect(voice.controller.getState().speechBackpressure).toMatchObject({
      queuedChunks: 2,
      maxQueuedChunks: 2
    });
    expect(
      voice.controller.getState().speechBackpressure.skippedCharacters
    ).toBe(0);
    expect(
      voice.controller.getState().speechBackpressure.pendingCharacters
    ).toBeGreaterThan(2 * 80);
    // Finishing each native utterance drains every coalesced sentence in
    // bounded chunks rather than replacing older speech with newer output.
    pending.resolve(result(fullReply.trim()));
    await vi.waitFor(() =>
      expect(voice.controller.getState().lastReply?.text).toBe(fullReply.trim())
    );
    for (let index = 0; index < 30; index += 1) {
      const handlers = voice.speechHandlers();
      handlers.onEnd();
      if (voice.controller.getState().phase === "listening") break;
    }
    expect(voice.controller.getState().speechBackpressure).toMatchObject({
      queuedChunks: 0,
      pendingCharacters: 0,
      skippedCharacters: 0
    });
    await voice.controller.stop();
  });

  it("buffers the full slow reply and disables microphone-triggered barge-in", async () => {
    const pending = deferred<ChatResult>();
    const voice = harness({
      send: vi.fn(() => pending.promise),
      preferences: preferences({ deliveryMode: "buffered", pace: "slow" })
    });
    await voice.controller.start();
    voice.handlers().onFinal({ transcript: "buffer this" });
    await vi.waitFor(() => expect(voice.chat.send).toHaveBeenCalledOnce());
    voice.emitStream(token("voice-utterance-1", 0, "Partial output. "));
    expect(voice.synthesis.speak).not.toHaveBeenCalled();

    voice.handlers().onSpeechStart();
    voice.handlers().onFinal({ transcript: "must not barge in" });
    expect(voice.chat.cancel).not.toHaveBeenCalled();
    expect(voice.chat.send).toHaveBeenCalledTimes(1);

    pending.resolve(result("The complete buffered reply."));
    await vi.waitFor(() => expect(voice.synthesis.speak).toHaveBeenCalledOnce());
    expect(voice.synthesis.speak).toHaveBeenCalledWith(
      "The complete buffered reply.",
      expect.any(Object),
      { rate: 0.8 }
    );
    await voice.controller.stop();
    expect(voice.chat.cancel).toHaveBeenCalledWith(
      "brain-one",
      "voice-utterance-1"
    );
  });

  it("re-arms buffered recognition from fresh interim speech when an engine omits speech-start", async () => {
    const voice = harness({
      preferences: preferences({ deliveryMode: "buffered" })
    });
    await voice.controller.start();
    voice.handlers().onFinal({ transcript: "first buffered request" });
    await vi.waitFor(() => expect(voice.synthesis.speak).toHaveBeenCalledOnce());
    voice.speechHandlers().onEnd();

    // A delayed final without a new speech signal is treated as buffered TTS
    // echo and cannot create another human turn.
    voice.handlers().onFinal({ transcript: "delayed speaker echo" });
    expect(voice.chat.send).toHaveBeenCalledTimes(1);

    // Some platform recognizers omit speech-start but still send interim text.
    voice.handlers().onInterim("new human words");
    voice.handlers().onFinal({ transcript: "new human words" });
    await vi.waitFor(() => expect(voice.chat.send).toHaveBeenCalledTimes(2));
  });

  it.each([
    ["neural listening", preferences({ neuralListening: true }), "neural-listening-unavailable"],
    ["neural voice", preferences({ neuralVoice: true }), "neural-voice-unavailable"]
  ] as const)("fails honestly when %s has no trained modality adapter", async (_label, setting, code) => {
    const voice = harness({ preferences: setting });
    await voice.controller.start();

    expect(voice.capture.request).not.toHaveBeenCalled();
    expect(voice.recognition.start).not.toHaveBeenCalled();
    expect(voice.synthesis.speak).not.toHaveBeenCalled();
    expect(voice.controller.getState()).toMatchObject({
      enabled: false,
      phase: "unavailable",
      error: { code }
    });
    expect(voice.controller.getState().error?.message).toContain(
      "was not substituted"
    );
  });

  it("routes selected neural listening through same-brain audio while retaining literal platform transcription", async () => {
    const neuralStop = vi.fn();
    const neural: LiveVoiceNeuralAdapter = {
      inspect: vi.fn(async () => ({
        listening: true,
        voice: false,
        detail:
          "Waveforms enter this brain's assemblies; Web Speech remains the transcript boundary."
      })),
      startListening: vi.fn(async () => ({ stop: neuralStop }))
    };
    const voice = harness({
      preferences: preferences({ neuralListening: true }),
      neural
    });

    await voice.controller.start();
    expect(neural.inspect).toHaveBeenCalledWith("brain-one");
    expect(neural.startListening).toHaveBeenCalledWith("brain-one");
    expect(voice.capture.request).not.toHaveBeenCalled();
    expect(voice.recognition.start).toHaveBeenCalledOnce();
    expect(voice.controller.getState().capabilities).toMatchObject({
      neuralListening: true,
      neuralVoice: false
    });

    voice.handlers().onFinal({ transcript: "literal recognized words" });
    await vi.waitFor(() =>
      expect(voice.chat.send).toHaveBeenCalledWith(
        "brain-one",
        "literal recognized words",
        "voice-utterance-1"
      )
    );
    await voice.controller.stop();
    expect(neuralStop).toHaveBeenCalledOnce();
  });

  it("uses neural speech only when a compatible adapter explicitly verifies it", async () => {
    const neuralSpeak = vi.fn((_text, handlers: LiveVoiceSynthesisHandlers) => {
      handlers.onStart();
      return { cancel: vi.fn() };
    });
    const neural: LiveVoiceNeuralAdapter = {
      inspect: vi.fn(async () => ({
        listening: false,
        voice: true,
        detail: "Verified intelligible neural speech fixture."
      })),
      startListening: vi.fn(async () => ({ stop: vi.fn() })),
      synthesis: {
        available: true,
        speak: neuralSpeak,
        cancel: vi.fn()
      }
    };
    const voice = harness({
      preferences: preferences({ neuralVoice: true }),
      neural
    });

    await voice.controller.start();
    voice.handlers().onFinal({ transcript: "speak through verified neural output" });
    await vi.waitFor(() => expect(neuralSpeak).toHaveBeenCalledOnce());
    expect(voice.synthesis.speak).not.toHaveBeenCalled();
    expect(neuralSpeak).toHaveBeenCalledWith(
      "A spoken brain reply.",
      expect.any(Object),
      { rate: 1 }
    );
  });

  it("closes a microphone stream that resolves after voice was stopped", async () => {
    const pendingCapture = deferred<{ stop(): void }>();
    const delayedStop = vi.fn();
    const voice = harness({ captureRequest: vi.fn(() => pendingCapture.promise) });
    const starting = voice.controller.start();
    await vi.waitFor(() => expect(voice.capture.request).toHaveBeenCalledOnce());
    await voice.controller.stop();
    pendingCapture.resolve({ stop: delayedStop });
    await starting;

    expect(delayedStop).toHaveBeenCalledOnce();
    expect(voice.recognition.start).not.toHaveBeenCalled();
    expect(voice.controller.getState().phase).toBe("idle");
  });
});
