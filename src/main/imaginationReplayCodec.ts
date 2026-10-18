import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, open, realpath } from "node:fs/promises";
import { join } from "node:path";
import type { ImaginationAssemblyManifestSummary, ImaginationReplaySpan } from "../shared/imaginationAssembly";
import type { NativeMediaTimeline } from "../shared/nativeMediaTimeline";
import { verifyNativeVideoRuntime, type PreparedVideoRuntime } from "./videoRuntimeProvisioner";
import type { VideoRuntimeTarget } from "./videoRuntimeManifest";

export interface ReplayProcessResult { exitCode: number; stdout: string; stderr: string; truncated: boolean; frame: number; outTimeUs: number }
export type ReplayRunner = (args: string[], cwd: string, signal: AbortSignal, progress?: (frame: number, outTimeUs: number) => void) => Promise<ReplayProcessResult>;
export interface ReplayCodecDependencies {
  runtime: PreparedVideoRuntime;
  checkResources(bytes: number, pids: readonly number[]): Promise<void>;
  reserve(bytes: number): Promise<void>;
  estimatedPeakBytes: number;
  timeoutMs?: number;
}
export interface InspectedReplaySource { path: string; mimeType: string; width?: number; height?: number; frames: number; durationMs: number; hasAudio: boolean; timeline?: NativeMediaTimeline }
const activeCodecPids=new Set<number>();
export async function replayFileHash(path: string): Promise<string> { const hash = createHash("sha256"); for await (const data of createReadStream(path)) hash.update(data); return hash.digest("hex"); }

/** Shell-free codec child with actual process-family cancellation and live resource checks. */
export async function createReplayRunner(deps: ReplayCodecDependencies): Promise<ReplayRunner> {
  const path = await realpath(deps.runtime.executablePath), info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("Replay codec must be an existing real native executable.");
  await verifyNativeVideoRuntime(path, (deps.runtime.target ?? `${process.platform}-${process.arch}`) as VideoRuntimeTarget);
  if (deps.runtime.binarySha256 && await replayFileHash(path) !== deps.runtime.binarySha256) throw new Error("Replay codec changed after reviewed runtime preparation.");
  return async (args, cwd, signal, progress) => {
    signal.throwIfAborted(); await deps.checkResources(deps.estimatedPeakBytes, [...activeCodecPids]); await deps.reserve(8 * 1024 * 1024);
    return new Promise<ReplayProcessResult>((resolveRun, rejectRun) => {
      const child = spawn(path, ["-nostdin", "-hide_banner", "-nostats", "-threads", "1", ...args, "-progress", "pipe:1"], { shell: false, windowsHide: true, detached: process.platform !== "win32", cwd, stdio: ["ignore", "pipe", "pipe"] });
      if(child.pid)activeCodecPids.add(child.pid);
      let failure: Error | undefined, stderr = "", stdout = "", line = "", truncated = false, frame = 0, outTimeUs = 0, checking = false, forced: NodeJS.Timeout | undefined;
      const terminate = (force = false) => { if (child.exitCode !== null) return;
        if (process.platform === "win32" && child.pid) { const killer = spawn("taskkill.exe", ["/PID", String(child.pid), "/T", "/F"], { shell: false, windowsHide: true, stdio: "ignore" }); killer.once("error", () => child.kill("SIGKILL")); }
        else try { if (!child.pid) throw new Error("Codec child has no process identity."); process.kill(-child.pid, force ? "SIGKILL" : "SIGTERM"); } catch { child.kill(force ? "SIGKILL" : "SIGTERM"); }
      };
      const stop = (error: unknown) => { if (failure) return; failure = error instanceof Error ? error : new Error(String(error)); terminate(); forced = setTimeout(() => terminate(true), 1000); forced.unref(); };
      const abort = () => stop(signal.reason ?? new Error("Replay composition cancelled.")); signal.addEventListener("abort", abort, { once: true }); if (signal.aborted) abort();
      const timeout = setTimeout(() => stop(new Error("Replay codec reached its bounded process deadline; committed segments remain.")), deps.timeoutMs ?? 120000); timeout.unref();
      const resources = setInterval(() => { if (checking || failure) return; checking = true; void deps.checkResources(deps.estimatedPeakBytes, [...activeCodecPids]).then(() => deps.reserve(8 * 1024 * 1024)).catch(stop).finally(() => { checking = false; }); }, 250); resources.unref();
      child.stdout.on("data", (data: Buffer) => { line += data.toString("utf8"); if (line.length > 65536) { stop(new Error("Codec progress exceeded the bounded line budget.")); return; }
        let end: number; while ((end = line.indexOf("\n")) >= 0) { const value = line.slice(0, end).trim(); line = line.slice(end + 1); const parts = value.split("=");
          if (parts[0] === "frame" && /^\d+$/.test(parts[1] ?? "")) frame = Math.max(frame, Number(parts[1]));
          if (parts[0] === "out_time_us" && /^\d+$/.test(parts[1] ?? "")) outTimeUs = Math.max(outTimeUs, Number(parts[1]));
          if (!/^[a-z_]+=/i.test(value)) { const room = 262144 - Buffer.byteLength(stdout), report = Buffer.from(`${value}\n`); if (report.byteLength > room) truncated = true; if (room > 0) stdout += report.subarray(0, room).toString("utf8"); }
          if (parts[0] === "progress") try { progress?.(frame, outTimeUs); } catch { /* Display callbacks do not own codec execution. */ }
        }
      });
      child.stderr.on("data", (data: Buffer) => { const text = data.toString("utf8"), available = 262144 - Buffer.byteLength(stderr); if (Buffer.byteLength(text) > available) truncated = true; if (available > 0) stderr += Buffer.from(text).subarray(0, available).toString("utf8"); });
      child.once("error", error => { failure ??= error; });
      child.once("close", code => { if(child.pid)activeCodecPids.delete(child.pid);clearTimeout(timeout); clearInterval(resources); if (forced) clearTimeout(forced); signal.removeEventListener("abort", abort);
        if (failure) rejectRun(failure); else resolveRun({ exitCode: code ?? 1, stdout, stderr, truncated, frame, outTimeUs }); });
    });
  };
}

function input(path: string, mime: string): string[] {
  const formats: Record<string, string> = { "image/png": "png_pipe", "image/apng": "apng", "audio/wav": "wav", "video/mp4": "mov" };
  const format = formats[mime]; if (!format) throw new Error("This retained media format has no supported replay decoder; original bytes remain saved.");
  return ["-protocol_whitelist", "file,pipe", "-f", format, ...(format === "apng" ? ["-ignore_loop", "1"] : []), "-i", path];
}
export async function inspectReplaySource(run: ReplayRunner, path: string, mimeType: string, timeline: NativeMediaTimeline | undefined, cwd: string, signal: AbortSignal): Promise<InspectedReplaySource> {
  const result = await run(["-loglevel", "info", ...input(path, mimeType), "-map", "0:v:0?", "-map", "0:a:0?", "-f", "null", "-"], cwd, signal);
  if (result.exitCode !== 0 || result.truncated) throw new Error(`Retained media did not pass bounded full decode: ${result.stderr.slice(-1800)}`);
  const dimensions = /Video:[^\r\n]*?\b(\d+)x(\d+)\b/.exec(result.stderr), width = dimensions ? Number(dimensions[1]) : undefined, height = dimensions ? Number(dimensions[2]) : undefined;
  const hasAudio = /Stream[^\r\n]*Audio:/.test(result.stderr), durationMs = result.outTimeUs / 1000;
  if (timeline?.unit === "frames" && result.frame !== timeline.count || timeline?.unit === "image" && result.frame !== 1 || timeline?.unit === "samples" && !hasAudio || timeline?.containerAudio && !hasAudio) throw new Error("Decoded media disagrees with its exact byte-producer timeline.");
  if (timeline && timeline.unit !== "image") {
    const expected = timeline.count * timeline.timebaseNumerator * 1000 / timeline.timebaseDenominator;
    if (Math.abs(durationMs - expected) > Math.max(2, 2 * timeline.timebaseNumerator * 1000 / timeline.timebaseDenominator, timeline.containerAudio ? 2048 * 1000 / timeline.containerAudio.sampleRate : 0)) throw new Error("Decoded duration disagrees with the native source range; no clock was fabricated.");
  }
  return { path, mimeType, ...(width ? { width, height } : {}), frames: result.frame, durationMs, hasAudio, timeline };
}
const seconds = (ms: number) => (ms / 1000).toFixed(9);
export async function renderReplaySpan(run: ReplayRunner, manifest: Pick<ImaginationAssemblyManifestSummary, "options" | "outputKind" | "trackCount">, span: ImaginationReplaySpan, sources: Map<string, InspectedReplaySource>, output: string, cwd: string, signal: AbortSignal, progress?: (frame: number) => void): Promise<void> {
  const { width, height, fps } = manifest.options, duration = span.endMs - span.startMs, columns = Math.ceil(Math.sqrt(Math.max(1, manifest.trackCount))), rows = Math.ceil(Math.max(1, manifest.trackCount) / columns), audioOnly = manifest.outputKind === "audio-only";
  const cellWidth = Math.max(2, Math.floor(width / columns / 2) * 2), cellHeight = Math.max(2, Math.floor(height / rows / 2) * 2);
  const args = ["-loglevel", "error", ...(audioOnly ? [] : ["-f", "lavfi", "-i", `color=c=black:s=${width}x${height}:r=${fps}:d=${seconds(duration)}`]), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"];
  const filters: string[] = audioOnly ? [] : ["[0:v]setpts=PTS-STARTPTS[background]"], audio: string[] = []; let previous = "background", nextInput = audioOnly ? 1 : 2;
  for (const state of span.states) {
    if (state.visual === "none" && state.audio === "none") continue; const source = sources.get(state.fragmentId); if (!source) throw new Error("A selected delivered fragment is not verified for composition.");
    args.push(...input(source.path, source.mimeType)); const index = nextInput++, label = `tile${index}`, nativeDuration = source.timeline ? source.timeline.count * source.timeline.timebaseNumerator * 1000 / source.timeline.timebaseDenominator : source.durationMs;
    const frameMs = source.timeline?.unit === "frames" ? source.timeline.timebaseNumerator * 1000 / source.timeline.timebaseDenominator : 0;
    const start = state.visual === "hold" ? Math.max(0, nativeDuration - frameMs) : state.sourceTimeMs ?? 0;
    if (state.visual !== "none") {
      const temporal = state.visual === "snapshot" ? `loop=loop=-1:size=1:start=0,trim=duration=${seconds(duration)}` : `trim=start=${seconds(start)}:end=${seconds(Math.min(nativeDuration, start + duration))},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=${seconds(duration)},trim=duration=${seconds(duration)}`;
      filters.push(`[${index}:v]${temporal},setpts=PTS-STARTPTS,scale=${cellWidth}:${cellHeight}:force_original_aspect_ratio=decrease,pad=${cellWidth}:${cellHeight}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps=${fps},format=yuv420p[${label}]`);
      const x = state.slot % columns * cellWidth, y = Math.floor(state.slot / columns) * cellHeight, combined = `canvas${index}`;
      filters.push(`[${previous}][${label}]overlay=x=${x}:y=${y}:shortest=1[${combined}]`); previous = combined;
    }
    if (state.audio === "play" && source.hasAudio) { const name = `sound${index}`; filters.push(`[${index}:a]atrim=start=${seconds(state.sourceTimeMs ?? 0)}:end=${seconds((state.sourceTimeMs ?? 0) + duration)},asetpts=PTS-STARTPTS,aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,apad=whole_dur=${seconds(duration)},atrim=duration=${seconds(duration)}[${name}]`); audio.push(name); }
  }
  if (audio.length) filters.push(`${audio.map(name => `[${name}]`).join("")}amix=inputs=${audio.length}:duration=longest:normalize=1,apad=whole_dur=${seconds(duration)},atrim=duration=${seconds(duration)}[mixed]`);
  else filters.push(`[${audioOnly ? 0 : 1}:a]atrim=duration=${seconds(duration)},asetpts=PTS-STARTPTS[mixed]`);
  args.push("-filter_complex_threads", "1", "-filter_complex", filters.join(";"), ...(audioOnly ? [] : ["-map", `[${previous}]`, "-frames:v", String(span.endFrame - span.startFrame), "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p"]), "-map", "[mixed]", "-t", seconds(duration), "-c:a", "pcm_s16le", "-n", output);
  const result = await run(args, cwd, signal, frame => progress?.(frame)); if (result.exitCode !== 0) throw new Error(`Delivered replay segment failed: ${result.stderr.slice(-1800)}`);
}
export async function joinReplaySpans(run: ReplayRunner, count: number, cwd: string, output: string, signal: AbortSignal, audioOnly: boolean, reserve: (bytes: number) => Promise<void>, clock: { fps: number; durationMs: number; spanDuration(index: number): number }): Promise<void> {
  const list = join(cwd, "segments.ffconcat"), file = await open(list, "wx", 0o600); try { await file.write("ffconcat version 1.0\n"); for (let i = 0; i < count; i++) { signal.throwIfAborted(); if (i % 256 === 0) await reserve(65536); await file.write(`file segment-${i}.${audioOnly ? "wav" : "mkv"}\nduration ${seconds(clock.spanDuration(i))}\n`); } await file.sync(); } finally { await file.close(); }
  const result = await run(["-loglevel", "error", "-protocol_whitelist", "file,pipe", "-f", "concat", "-safe", "1", "-i", list, ...(audioOnly ? ["-c:a", "pcm_s16le"] : ["-r",String(clock.fps),"-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart"]), "-t",seconds(clock.durationMs),"-n", output], cwd, signal);
  if (result.exitCode !== 0) throw new Error(`Delivered replay assembly failed: ${result.stderr.slice(-1800)}`);
}

export async function requireReplayEncoders(run: ReplayRunner, cwd: string, signal: AbortSignal, audioOnly: boolean): Promise<string[]> {
  const result = await run(["-encoders"], cwd, signal); if (result.exitCode !== 0 || result.truncated) throw new Error("Selected runtime did not provide a complete bounded encoder report.");
  const available = new Set([...result.stdout.matchAll(/^\s*[VAS][A-Z.]{5}\s+(\S+)\s/gm)].map(match => match[1]!));
  const required = audioOnly ? ["pcm_s16le"] : ["ffv1", "libx264", "aac", "pcm_s16le"];
  for (const encoder of required) if (!available.has(encoder)) throw new Error(`The selected licensed runtime lacks ${encoder}; no other codec/runtime was silently substituted.`);
  return required;
}
