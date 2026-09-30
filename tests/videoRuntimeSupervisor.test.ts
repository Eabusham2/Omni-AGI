import { describe, expect, it, vi } from "vitest";
import { EngineSupervisor, requestNeedsVideoRuntime } from "../src/main/engineSupervisor";
import { VideoRuntimeUnavailableError, type PreparedVideoRuntime } from "../src/main/videoRuntimeProvisioner";

function supervisor(prepare: (signal: AbortSignal) => Promise<PreparedVideoRuntime>) {
  const engine = new EngineSupervisor({
    appPath: "/fixture/not-a-real-app", videoRuntimeCacheRoot: "/fixture/cache", prepareVideoRuntime: prepare
  });
  // These tests never start a process, create/load a brain, or run a model.
  const start = vi.spyOn(engine, "start").mockResolvedValue(true);
  const raw = vi.spyOn(engine as unknown as {
    rawRequest(method: string, params: Record<string, unknown>, timeoutMs: number, signal?: AbortSignal): Promise<unknown>;
  }, "rawRequest").mockResolvedValue({ fixture: true });
  return { engine, start, raw };
}

describe("codec-need-only video runtime preparation", () => {
  it("does not select health, create/load, text training/chat, image generation or raw PCM packets", async () => {
    const prepare = vi.fn(async () => ({ state: "external", executablePath: "/fixture/ffmpeg" } as PreparedVideoRuntime));
    const { engine } = supervisor(prepare);
    for (const method of ["health", "create", "load", "train", "chat", "modality_capabilities", "start_observation"]) {
      expect(requestNeedsVideoRuntime(method, { modalities: ["video"], modality: "video" })).toBe(false);
      await engine.request(method);
    }
    expect(requestNeedsVideoRuntime("generate_modality", { modality: "image" })).toBe(false);
    expect(requestNeedsVideoRuntime("ingest", { path: "/fixture/file.txt", kind: "text" })).toBe(false);
    expect(requestNeedsVideoRuntime("ingest", { path: "/fixture/video.gif", kind: "video" })).toBe(false);
    expect(requestNeedsVideoRuntime("ingest", { path: "/fixture/audio.wav", kind: "audio" })).toBe(false);
    expect(requestNeedsVideoRuntime("observe_packet", { mimeType: "audio/wav" })).toBe(false);
    expect(requestNeedsVideoRuntime("observe_packet", { mimeType: "audio/pcm" })).toBe(false);
    expect(prepare).not.toHaveBeenCalled();
  });

  it("identifies common media ingest, encoded packets and video generation", () => {
    expect(requestNeedsVideoRuntime("generate_modality", { modality: "video" })).toBe(true);
    expect(requestNeedsVideoRuntime("ingest", { path: "/fixture/video.MP4" })).toBe(true);
    expect(requestNeedsVideoRuntime("ingest", { kind: "audio" })).toBe(true);
    expect(requestNeedsVideoRuntime("observe_packet", { mimeType: "audio/webm;codecs=opus" })).toBe(true);
  });

  it("configures an already-warm worker before the media call without restarting or loading a brain", async () => {
    const prepare = vi.fn(async () => ({
      state: "ready", executablePath: "/fixture/cache/pinned/ffmpeg", artifactSha256: "a".repeat(64),
      binarySha256: "b".repeat(64), binarySizeBytes: 128, target: "linux-x64"
    } as PreparedVideoRuntime));
    const { engine, raw } = supervisor(prepare);
    await engine.request("generate_modality", { brainId: "fixture", jobId: "fixture-job", modality: "video" });
    expect(raw.mock.calls.map((call) => call[0])).toEqual(["configure_video_runtime", "generate_modality"]);
    expect(raw.mock.calls[0]?.[1]).toMatchObject({ binarySha256: "b".repeat(64), target: "linux-x64" });
    expect(prepare).toHaveBeenCalledOnce();
    expect(raw.mock.calls.some((call) => ["reload", "load", "create", "shutdown"].includes(call[0]))).toBe(false);
  });

  it("reports the missing vetted catalog without disabling existing media fallback", async () => {
    const prepare = vi.fn(async () => { throw new VideoRuntimeUnavailableError("linux-x64", "Fixture missing reviewed builds"); });
    const { engine, raw } = supervisor(prepare);
    const diagnostics: string[] = [];
    engine.on("diagnostic", (message: string) => diagnostics.push(message));
    await engine.request("generate_modality", { modality: "video" });
    expect(raw.mock.calls.map((call) => call[0])).toEqual(["generate_modality"]);
    expect(diagnostics).toEqual([expect.stringContaining("missing reviewed builds")]);
  });

  it("does not bypass an invalid binary or cancellation to start a media call", async () => {
    const bad = supervisor(async () => { throw new Error("Fixture binary hash mismatch"); });
    await expect(bad.engine.request("generate_modality", { modality: "video" })).rejects.toThrow("hash mismatch");
    expect(bad.raw).not.toHaveBeenCalled();
    const controller = new AbortController();
    const prepare = vi.fn(async (signal: AbortSignal) => new Promise<PreparedVideoRuntime>((_resolve, reject) => {
      signal.addEventListener("abort", () => reject(new Error("Video runtime setup was cancelled.")), { once: true });
    }));
    const stopped = supervisor(prepare);
    const operation = stopped.engine.request("generate_modality", { modality: "video" }, 1_000, controller.signal);
    const rejection = expect(operation).rejects.toThrow("cancelled");
    await vi.waitFor(() => expect(prepare).toHaveBeenCalledOnce());
    controller.abort();
    await rejection;
    expect(stopped.raw).not.toHaveBeenCalled();
  });

  it("does not expose the worker runtime path setter through public request forwarding", async () => {
    const { engine, raw } = supervisor(async () => ({ state: "external", executablePath: "/fixture/ffmpeg" }));
    await expect(engine.request("configure_video_runtime", { executablePath: "/arbitrary/script" })).rejects.toThrow("internal verified-main");
    expect(raw).not.toHaveBeenCalled();
  });
});
