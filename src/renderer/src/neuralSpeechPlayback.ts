import type { LiveVoiceSynthesisAdapter, LiveVoiceSynthesisHandlers } from "../../shared/liveVoice";
import type { OmniApi, RuntimeJob } from "../../shared/types";

export interface NeuralWavePlayer {
  playbackRate: number;
  volume: number;
  src: string;
  onplaying: (() => void) | null;
  onended: (() => void) | null;
  onerror: (() => void) | null;
  play(): Promise<void>;
  pause(): void;
}

export type NeuralWavePlayerConstructor = new (src: string) => NeuralWavePlayer;
const record = (value: unknown): Record<string, unknown> | undefined =>
  typeof value === "object" && value !== null ? value as Record<string, unknown> : undefined;

/** Same-brain generated WAV only; never calls platform TTS or an external model. */
export function createNeuralSpeechSynthesisAdapter(
  brainId: () => string | undefined,
  api: OmniApi["modality"] | undefined,
  Player: NeuralWavePlayerConstructor | undefined
): LiveVoiceSynthesisAdapter {
  const active = new Set<() => void>();
  return {
    available: Boolean(Player && typeof api?.generateSpeech === "function" &&
      typeof api?.onGeneration === "function" && typeof api?.cancel === "function" && globalThis.crypto?.subtle),
    deferredStart: true,
    cancel: () => { for (const cancel of [...active]) cancel(); },
    speak(text: string, handlers: LiveVoiceSynthesisHandlers, options) {
      const owner = brainId();
      if (!owner || !api || !Player) throw new Error("Same-brain waveform output or local playback is unavailable.");
      const requestId = globalThis.crypto?.randomUUID?.() ?? `speech-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      let cancelled = false;
      let finished = false;
      let accepted = false;
      let jobId: string | undefined;
      let player: NeuralWavePlayer | undefined;
      let unsubscribe: (() => void) | undefined;
      const cleanup = (): void => {
        unsubscribe?.(); unsubscribe = undefined;
        active.delete(cancel);
        if (player) { player.onplaying = null; player.onended = null; player.onerror = null; }
      };
      const fail = (error: unknown): void => {
        if (cancelled || finished) return;
        finished = true;
        cleanup();
        player?.pause();
        handlers.onError(error instanceof Error ? error.message : String(error));
      };
      const cancel = (): void => {
        if (cancelled || finished) return;
        cancelled = true;
        player?.pause();
        if (player) player.src = "";
        cleanup();
        if (jobId && !accepted) void api.cancel(jobId).catch(() => undefined);
      };
      const receive = (job: RuntimeJob): void => {
        if (job.brainId !== owner || job.kind !== "audio" ||
            (job.speechRequestId !== requestId && job.id !== jobId)) return;
        jobId = job.id;
        if (cancelled) {
          if (!["complete", "cancelled", "failed"].includes(job.state)) void api.cancel(job.id).catch(() => undefined);
          return;
        }
        if (accepted || finished) return;
        if (job.state === "failed" || job.state === "cancelled") {
          fail(new Error(job.error ?? "Same-brain waveform generation did not complete.")); return;
        }
        if (job.state !== "complete") return;
        const output = record(job.output), speech = record(output?.speech);
        const url = typeof output?.mediaUrl === "string" ? output.mediaUrl : output?.dataUrl;
        if (output?.brainId !== owner || output?.modality !== "audio" || output?.mimeType !== "audio/wav" ||
            speech?.requestId !== requestId || speech.source !== "same-brain-audio-region" ||
            speech.sameBrain !== true || speech.externalModelUsed !== false ||
            speech.intelligibilityVerified !== false || typeof url !== "string" ||
            !/^(?:omni-media:|data:audio\/wav;base64,)/.test(url)) {
          fail(new Error("The generated waveform is not bound to this same-brain voice request.")); return;
        }
        accepted = true;
        unsubscribe?.(); unsubscribe = undefined;
        void textDigest.then((digest) => {
          if (cancelled || finished) return;
          if (speech.textSha256 !== digest || speech.textConditioned !== true) {
            fail(new Error("The waveform text hash does not match this exact reply.")); return;
          }
          try {
            player = new Player(url);
            player.playbackRate = Math.max(0.1, Math.min(10, options.rate));
            player.volume = 0.4;
            player.onplaying = () => { if (!cancelled && !finished) handlers.onStart(); };
            player.onended = () => {
              if (!cancelled && !finished) { finished = true; cleanup(); handlers.onEnd(); }
            };
            player.onerror = () => fail(new Error("Same-brain WAV playback failed."));
            void player.play().catch(fail);
          } catch (error) { fail(error); }
        }).catch(fail);
      };
      active.add(cancel);
      let textDigest: Promise<string>;
      try {
        textDigest = globalThis.crypto.subtle.digest("SHA-256", new TextEncoder().encode(text))
          .then((digest) => Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join(""));
        // Digest rejection is also observed before a producer completion arrives.
        void textDigest.catch(fail);
        unsubscribe = api.onGeneration(({ job }) => receive(job));
        void api.generateSpeech({ brainId: owner, requestId, text, rate: options.rate }).then((job) => {
          if (!job || job.brainId !== owner || job.kind !== "audio" || job.speechRequestId !== requestId) {
            fail(new Error("The voice job is not bound to this brain and request.")); return;
          }
          jobId ??= job.id;
          receive(job);
        }).catch(fail);
      } catch (error) { fail(error); }
      return { cancel };
    }
  };
}
