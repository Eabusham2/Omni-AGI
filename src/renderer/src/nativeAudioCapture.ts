import type { LiveVoiceCaptureAdapter, LiveVoiceCaptureSession, LiveVoiceNativeAudioHandlers } from "../../shared/liveVoice";

export interface NativeAudioCaptureAuthorization {
  sessionId: string;
  maxBytes: number;
  stop(): Promise<unknown>;
}
export interface NativeAudioCaptureAdapter {
  available: boolean;
  start(authorize: (sampleRate: number) => Promise<NativeAudioCaptureAuthorization>, handlers: LiveVoiceNativeAudioHandlers, signal?: AbortSignal): Promise<LiveVoiceCaptureSession>;
}

/** Real PCM encoding only. No recognizer, transcript, model, or generated data. */
export function pcmWave(samples: readonly Float32Array[], sampleRate: number): Uint8Array {
  if (!Number.isSafeInteger(sampleRate) || sampleRate < 1) throw new Error("Microphone sample rate is invalid.");
  const count = samples.reduce((sum, value) => sum + value.length, 0);
  if (!Number.isSafeInteger(count) || count < 1 || count * 2 + 36 > 0xffffffff) throw new Error("Microphone WAV needs a supported RIFF size; audio was not truncated.");
  const result = new Uint8Array(44 + count * 2), view = new DataView(result.buffer);
  for (const [position, text] of [[0, "RIFF"], [8, "WAVE"], [12, "fmt "], [36, "data"]] as const)
    for (let index = 0; index < text.length; index += 1) result[position + index] = text.charCodeAt(index);
  view.setUint32(4, result.length - 8, true); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); view.setUint32(40, count * 2, true);
  let offset = 44;
  for (const block of samples) for (const value of block) {
    if (!Number.isFinite(value)) throw new Error("Microphone samples contain a nonfinite signal.");
    const clipped = Math.max(-1, Math.min(1, value));
    view.setInt16(offset, Math.round(clipped * (clipped < 0 ? 32768 : 32767)), true); offset += 2;
  }
  return result;
}

function base64(data: Uint8Array): string {
  let text = "";
  for (let offset = 0; offset < data.length; offset += 8192) text += String.fromCharCode(...data.subarray(offset, offset + 8192));
  return btoa(text);
}

function abortable<T>(promise: Promise<T>, signal: AbortSignal | undefined, closeLate: (value: T) => void): Promise<T> {
  if (!signal) return promise;
  return new Promise((resolve, reject) => {
    let ended = false;
    const abort = (): void => { if (ended) return; ended = true; reject(signal.reason ?? new DOMException("Audio capture stopped.", "AbortError")); };
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
    void promise.then(value => {
      signal.removeEventListener("abort", abort);
      if (ended) { closeLate(value); return; }
      ended = true; resolve(value);
    }, error => { signal.removeEventListener("abort", abort); if (!ended) { ended = true; reject(error); } });
  });
}

/** Silence segmentation is a capture boundary, not linguistic transcription. */
export function createNativeAudioCaptureAdapter(
  capture: LiveVoiceCaptureAdapter,
  Context: typeof AudioContext | undefined = typeof globalThis.AudioContext === "function" ? globalThis.AudioContext : undefined
): NativeAudioCaptureAdapter {
  return { available: capture.available && Boolean(Context), async start(authorize, handlers, signal) {
    if (!Context) throw new Error("Direct PCM microphone capture is unavailable; STT was not substituted.");
    signal?.throwIfAborted();
    const session = await abortable(capture.request(), signal, value => value.stop());
    let context: AudioContext | undefined, authorization: NativeAudioCaptureAuthorization | undefined;
    let source: MediaStreamAudioSourceNode | undefined, processor: ScriptProcessorNode | undefined, gain: GainNode | undefined;
    let stopped = false;
    const stop = (): void => {
      if (stopped) return; stopped = true;
      signal?.removeEventListener("abort", stop);
      if (processor) processor.onaudioprocess = null;
      processor?.disconnect(); source?.disconnect(); gain?.disconnect(); session.stop();
      void context?.close().catch(() => undefined);
      void authorization?.stop().catch(() => undefined);
    };
    signal?.addEventListener("abort", stop, { once: true });
    try {
      signal?.throwIfAborted();
      if (!session.stream) throw new Error("Direct microphone capture did not expose its actual stream.");
      context = new Context();
      source = context.createMediaStreamSource(session.stream);
      processor = context.createScriptProcessor(4096, 1, 1);
      gain = context.createGain(); gain.gain.value = 0;
      authorization = await abortable(authorize(context.sampleRate), signal, value => { void value.stop().catch(() => undefined); });
      if (stopped) throw signal?.reason ?? new DOMException("Audio capture stopped.", "AbortError");
      const permission = authorization;
      let recent: Float32Array[] = [], blocks: Float32Array[] = [], samples = 0, recentSamples = 0;
      let active = false, quiet = 0, noise = .002, sequence = 0, capturedAt = "";
      const rate = context.sampleRate;
      const finalize = (): void => {
        if (!active || !blocks.length || stopped) return;
        const data = pcmWave(blocks, rate);
        blocks = []; samples = 0; active = false; quiet = 0;
        handlers.onFinalAudio({ format: "omni-native-audio-chat", version: 1, captureSessionId: permission.sessionId,
          utteranceId: crypto.randomUUID(), sequence: sequence++, mimeType: "audio/wav", dataBase64: base64(data),
          capturedAt, sampleRate: rate, channels: 1 });
      };
      processor.onaudioprocess = (event) => {
        if (stopped) return;
        try {
          const values = event.inputBuffer.getChannelData(0);
          let energy = 0;
          for (const value of values) { if (!Number.isFinite(value)) throw new Error("Microphone PCM is nonfinite."); energy += value * value; }
          const rms = Math.sqrt(energy / Math.max(1, values.length)), speech = rms > Math.max(.008, noise * 3);
          if (!active && !speech) noise = .98 * noise + .02 * rms;
          if (!active && speech) {
            active = true; capturedAt = new Date().toISOString(); blocks = recent; samples = recentSamples;
            recent = []; recentSamples = 0; handlers.onSpeechStart();
          }
          const copy = values.slice();
          if (active) {
            if ((samples + copy.length) * 2 + 44 > permission.maxBytes) throw new Error("This utterance exceeds the current microphone/RAM transport envelope. Capture paused without sending truncated audio.");
            blocks.push(copy); samples += copy.length;
            quiet = speech ? 0 : quiet + copy.length;
            if (quiet >= rate * .65) finalize();
          } else {
            recent.push(copy); recentSamples += copy.length;
            while (recentSamples > rate * .2 && recent.length > 1) recentSamples -= recent.shift()!.length;
          }
        } catch (error) { stop(); handlers.onError(error); }
      };
      source.connect(processor); processor.connect(gain); gain.connect(context.destination);
      await abortable(context.resume(), signal, () => undefined);
      return { stop };
    } catch (error) { stop(); throw error; }
  } };
}
