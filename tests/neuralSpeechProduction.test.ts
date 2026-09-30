import { createHash, webcrypto } from "node:crypto";
import { EventEmitter } from "node:events";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { LiveVoiceSynthesisHandlers } from "../src/shared/liveVoice";
import type { NeuralSpeechGenerateRequest, OmniApi, RuntimeJob, RuntimeJobEvent } from "../src/shared/types";
import { createNeuralSpeechSynthesisAdapter, type NeuralWavePlayer } from "../src/renderer/src/neuralSpeechPlayback";
import { createOmniNeuralAudioAdapter } from "../src/renderer/src/browserLiveVoice";
import { RuntimeJobManager, persistedModalityCapabilities, type BrainService } from "../src/main/brainService";
import type { EngineSupervisor } from "../src/main/engineSupervisor";

// These are transport/player stubs. No neural model, microphone, codec or app is run.
class WavePlayer implements NeuralWavePlayer {
  static instances: WavePlayer[] = [];
  playbackRate = 1;
  volume = 1;
  onplaying: (() => void) | null = null;
  onended: (() => void) | null = null;
  onerror: (() => void) | null = null;
  play = vi.fn(async () => undefined);
  pause = vi.fn();
  constructor(public src: string) { WavePlayer.instances.push(this); }
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

const digest = (text: string) => createHash("sha256").update(text, "utf8").digest("hex");
function waveform(request: NeuralSpeechGenerateRequest, overrides: Record<string, unknown> = {}) {
  return { brainId: request.brainId, modality: "audio", mimeType: "audio/wav",
    mediaUrl: "omni-media://artifact/owned-waveform", speech: {
      requestId: request.requestId, source: "same-brain-audio-region", textSha256: digest(request.text),
      textConditioned: true, sameBrain: true, externalModelUsed: false, intelligibilityVerified: false,
      trainingState: "needs-speech-training", ...overrides
    } };
}
function runtimeJob(request: NeuralSpeechGenerateRequest, state: RuntimeJob["state"] = "running"): RuntimeJob {
  return { id: "wave-job", brainId: request.brainId, speechRequestId: request.requestId, kind: "audio", state,
    progress: state === "complete" ? 1 : 0, label: "Waveform fixture", createdAt: "2026-09-30T00:00:00.000Z",
    updatedAt: "2026-09-30T00:00:00.000Z", ...(state === "complete" ? { output: waveform(request) } : {}) };
}
function playbackFixture() {
  let listener: ((event: RuntimeJobEvent) => void) | undefined;
  let request!: NeuralSpeechGenerateRequest;
  const pending = deferred<RuntimeJob>();
  const unsubscribe = vi.fn(() => { listener = undefined; });
  const api = {
    generateSpeech: vi.fn((value: NeuralSpeechGenerateRequest) => { request = value; return pending.promise; }),
    onGeneration: vi.fn((value: typeof listener) => { listener = value; return unsubscribe; }),
    cancel: vi.fn(async () => runtimeJob(request, "cancelled"))
  } as unknown as OmniApi["modality"];
  const handlers: LiveVoiceSynthesisHandlers = { onStart: vi.fn(), onEnd: vi.fn(), onError: vi.fn() };
  const adapter = createNeuralSpeechSynthesisAdapter(() => "mind", api, WavePlayer);
  const session = adapter.speak(" exact reply \n", handlers, { rate: 1.35 });
  return { api, handlers, adapter, session, pending, unsubscribe, request: () => request,
    emit: (job: RuntimeJob) => listener?.({ job } as RuntimeJobEvent) };
}

describe("same-brain own waveform playback production route", () => {
  beforeEach(() => { vi.stubGlobal("crypto", webcrypto); WavePlayer.instances = []; });
  afterEach(() => { vi.unstubAllGlobals(); });

  it("plays only an exact owned text-conditioned WAV and waits for actual playback", async () => {
    const f = playbackFixture();
    expect(f.adapter.available).toBe(true);
    expect(f.request()).toMatchObject({ brainId: "mind", text: " exact reply \n", rate: 1.35 });
    expect(f.handlers.onStart).not.toHaveBeenCalled();
    f.pending.resolve(runtimeJob(f.request()));
    f.emit(runtimeJob(f.request(), "complete"));
    await vi.waitFor(() => expect(WavePlayer.instances).toHaveLength(1));
    const player = WavePlayer.instances[0]!;
    expect(player.src).toBe("omni-media://artifact/owned-waveform");
    expect(player.playbackRate).toBe(1.35);
    expect(player.volume).toBeLessThan(1);
    expect(f.handlers.onStart).not.toHaveBeenCalled();
    player.onplaying?.();
    expect(f.handlers.onStart).toHaveBeenCalledOnce();
    player.onended?.();
    expect(f.handlers.onEnd).toHaveBeenCalledOnce();
    expect(f.unsubscribe).toHaveBeenCalledOnce();
  });

  it("handles completion before the initial IPC snapshot without double playback", async () => {
    const f = playbackFixture();
    f.emit(runtimeJob(f.request(), "complete"));
    f.emit(runtimeJob(f.request(), "complete"));
    f.pending.resolve(runtimeJob(f.request()));
    await vi.waitFor(() => expect(WavePlayer.instances).toHaveLength(1));
    f.session.cancel();
    expect(WavePlayer.instances[0]!.pause).toHaveBeenCalledOnce();
    expect(WavePlayer.instances[0]!.src).toBe("");
    expect(f.api.cancel).not.toHaveBeenCalled(); // producer already completed; stop local audio only
  });

  it("cancels an exact late-created job before playback without cancelling text or another mind", async () => {
    const f = playbackFixture();
    f.session.cancel();
    f.emit({ ...runtimeJob(f.request(), "complete"), brainId: "another-mind" });
    f.pending.resolve(runtimeJob(f.request()));
    await vi.waitFor(() => expect(f.api.cancel).toHaveBeenCalledWith("wave-job"));
    expect(WavePlayer.instances).toHaveLength(0);
    expect(f.handlers.onStart).not.toHaveBeenCalled();
    expect(f.handlers.onError).not.toHaveBeenCalled();
  });

  it.each(["wrong-hash", "claimed-intelligibility"])("rejects %s metadata instead of playing or calling TTS", async (mode) => {
    const f = playbackFixture();
    const overrides = mode === "wrong-hash" ? { textSha256: digest("different text") } : { intelligibilityVerified: true };
    f.pending.resolve({ ...runtimeJob(f.request(), "complete"), output: waveform(f.request(), overrides) });
    await vi.waitFor(() => expect(f.handlers.onError).toHaveBeenCalledOnce());
    expect(WavePlayer.instances).toHaveLength(0);
  });

  it("ignores another brain or stale request's completion", async () => {
    const f = playbackFixture();
    f.emit({ ...runtimeJob(f.request(), "complete"), brainId: "another-mind" });
    f.emit({ ...runtimeJob(f.request(), "complete"), id: "another-job", speechRequestId: "stale-request" });
    await Promise.resolve();
    expect(WavePlayer.instances).toHaveLength(0);
    f.session.cancel();
    f.pending.resolve(runtimeJob(f.request(), "cancelled"));
  });

  it("exposes physical output without requiring trained input, ASR or verified intelligibility", async () => {
    const capabilities = vi.fn(async () => ({ brainId: "mind", audioPerception: false,
      audioGeneration: false, audioRegionAvailable: true, neuralSpeechSynthesis: false,
      speechPairedExamples: 0, speechQuality: "needs-speech-training" }));
    const api = { capabilities, generateSpeech: vi.fn(), onGeneration: vi.fn(), cancel: vi.fn() } as unknown as OmniApi["modality"];
    const adapter = createOmniNeuralAudioAdapter(api, { capabilities: { microphone: false, camera: false, screen: false }, open: vi.fn() },
      { brainId: "mind", audioConstructor: WavePlayer });
    await expect(adapter.inspect("mind")).resolves.toMatchObject({ listening: false, voice: true, voiceQuality: "needs-speech-training" });
    expect(adapter.synthesis?.available).toBe(true);
    await expect(adapter.inspect("other")).rejects.toThrow(/another mind/);
  });
});

function jobManager(engineOutput: (params: Record<string, unknown>) => Promise<unknown>) {
  const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
    if (method === "cancel") return { jobId: params.jobId, cancelled: true, acknowledged: true };
    return engineOutput(params);
  });
  const cancelRequest = vi.fn(async (requestId: string) => ({ requestId, acknowledged: true, phase: "running", workerTerminationAcknowledged: false }));
  const engine = Object.assign(new EventEmitter(), { request, cancelRequest, interruptAndRestart: vi.fn() }) as unknown as EngineSupervisor;
  const service = { preflightStart: vi.fn(async () => undefined), repository: { brainDirectory: () => "/fixture/mind" } } as unknown as BrainService;
  return { manager: new RuntimeJobManager(service, engine), request, cancelRequest, engine };
}

describe("native speech job main route without a neural worker", () => {
  it("routes literal text through native speech RPC and validates its exact digest", async () => {
    const f = jobManager(async (params) => waveform({ brainId: String(params.brainId), requestId: String(params.speechRequestId), text: String(params.text) }));
    const text = ` ${"long reply ".repeat(10_001)}\n`;
    const job = f.manager.generateSpeech({ brainId: "mind", requestId: "owned", text, rate: 1.35 });
    await vi.waitFor(() => expect(f.manager.list()[0]!.state).toBe("complete"));
    expect(f.request).toHaveBeenCalledWith("generate_neural_speech", expect.objectContaining({ jobId: job.id,
      brainId: "mind", text, rate: 1.35, speechRequestId: "owned", storagePath: "/fixture/mind" }), 0, expect.any(AbortSignal));
    expect(f.manager.list()[0]!.speechRequestId).toBe("owned");
  });

  it("fails a waveform bound to another text instead of advertising completion", async () => {
    const f = jobManager(async () => waveform({ brainId: "mind", requestId: "owned", text: "wrong" }));
    f.manager.generateSpeech({ brainId: "mind", requestId: "owned", text: "actual" });
    await vi.waitFor(() => expect(f.manager.list()[0]!.state).toBe("failed"));
    expect(f.manager.list()[0]!.error).toMatch(/exact same-brain speech request/);
  });

  it("accepts cooperative speech cancellation without terminating the warm worker", async () => {
    const pending = deferred<unknown>();
    const f = jobManager(() => pending.promise);
    const job = f.manager.generateSpeech({ brainId: "mind", requestId: "owned", text: "actual" });
    await vi.waitFor(() => expect(f.request).toHaveBeenCalledOnce());
    const cancelling = f.manager.cancel(job.id);
    pending.resolve(waveform({ brainId: "mind", requestId: "owned", text: "actual" }));
    await expect(cancelling).resolves.toMatchObject({ id: job.id, state: "cancelled" });
    expect(f.cancelRequest).toHaveBeenCalledWith(job.id);
    expect(f.engine.interruptAndRestart).not.toHaveBeenCalled();
    expect(f.manager.list()[0]!.output).toBeUndefined();
  });

  it.each([0, 2])("reads %i paired examples without claiming verified speech or loading a brain", async (pairs) => {
    const root = await mkdtemp(join(tmpdir(), "omni-speech-caps-"));
    try {
      await mkdir(join(root, "engine"));
      await writeFile(join(root, "engine", "brain.json"), JSON.stringify({ brain_id: "mind",
        config: { audio_enabled: true }, modality_training: { audio: 7, audio_speech_pairs: pairs } }));
      await expect(persistedModalityCapabilities(root, "mind")).resolves.toMatchObject({ audioRegionAvailable: true,
        audioGeneration: true, speechPairedExamples: pairs, speechQuality: pairs ? "unverified" : "needs-speech-training",
        neuralSpeechRecognition: false, neuralSpeechSynthesis: false });
    } finally { await rm(root, { recursive: true, force: true }); }
  });
});
