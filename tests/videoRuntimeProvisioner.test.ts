import { createHash } from "node:crypto";
import { chmod, mkdtemp, readFile, readdir, realpath, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DiskReservePauseError, MANDATORY_FREE_DISK_BYTES } from "../src/main/diskSpace";
import {
  bundledVideoRuntimeManifest, validateVideoRuntimeManifest,
  VIDEO_RUNTIME_TARGETS, videoRuntimeFiles,
  type VideoRuntimeArtifact, type VideoRuntimeFilePin, type VideoRuntimeTarget
} from "../src/main/videoRuntimeManifest";
import {
  VideoRuntimeProvisioner, VideoRuntimeUnavailableError, verifyNativeVideoRuntime,
  type VideoRuntimeProgress
} from "../src/main/videoRuntimeProvisioner";

const roots: string[] = [];
const releaseUrl = "https://github.com/Eabusham2/Omni-AGI/releases/download/video-runtime-fixture-v1/";
const hash = (bytes: Buffer): string => createHash("sha256").update(bytes).digest("hex");
const ampleDisk = async () => ({ diskTotalBytes: 256 * 1024 ** 3, diskFreeBytes: 100 * 1024 ** 3 });

function nativeBytes(target: VideoRuntimeTarget): Buffer {
  const bytes = Buffer.alloc(128);
  if (target.startsWith("linux-")) {
    bytes.set([0x7f, 0x45, 0x4c, 0x46, 2, 1]);
    bytes.writeUInt16LE(target.endsWith("-x64") ? 62 : 183, 18);
  } else if (target.startsWith("darwin-")) {
    bytes.writeUInt32LE(0xfeedfacf, 0);
    bytes.writeUInt32LE(target.endsWith("-x64") ? 0x01000007 : 0x0100000c, 4);
  } else {
    bytes.set([0x4d, 0x5a]);
    bytes.writeUInt32LE(80, 60);
    bytes.writeUInt32LE(0x00004550, 80);
    bytes.writeUInt16LE(target.endsWith("-x64") ? 0x8664 : 0xaa64, 84);
  }
  return bytes;
}

function fixture(target: VideoRuntimeTarget = "linux-x64") {
  const payloads = new Map<string, Buffer>();
  const pin = (fileName: string, payload: Buffer): VideoRuntimeFilePin => {
    const url = releaseUrl + fileName;
    payloads.set(url, payload);
    return { fileName, url, sizeBytes: payload.length, sha256: hash(payload) };
  };
  // These are fake metadata/native headers, not executable third-party code or
  // a real license review. Tests never run them, fetch a URL or train a brain.
  const artifact: VideoRuntimeArtifact = {
    id: `ffmpeg-fixture-${target}`, target, version: "9.0.2", license: "LGPL-2.1-or-later",
    binary: pin(target.startsWith("win32-") ? "ffmpeg.exe" : "ffmpeg", nativeBytes(target)),
    correspondingSource: pin("complete-sources.tar.xz", Buffer.from("fixture source")),
    buildAndInstallMaterial: pin("build-install.tar.xz", Buffer.from("fixture recipe data; never executed")),
    licenseNotices: pin("notices.txt", Buffer.from("fixture notice; not a reviewed real runtime")),
    provenance: {
      upstreamReleaseUrl: "https://ffmpeg.org/releases/ffmpeg-9.0.2.tar.xz",
      buildRepositoryUrl: "https://github.com/Eabusham2/Omni-AGI",
      buildCommit: "a".repeat(40), configuration: "--disable-autodetect --disable-gpl --disable-nonfree",
      linkage: "standalone-cli-static-except-system", reviewId: "fixture-only-not-a-legal-review",
      reviewedAt: "2026-09-29T00:00:00Z", completeCorrespondingSourceReviewed: true,
      allLinkedNonSystemLibrariesReviewed: true, distributionLicenseReviewed: true
    }, linkedLibraries: []
  };
  const manifest = validateVideoRuntimeManifest({
    schemaVersion: 1, component: "FFmpeg", artifacts: [artifact],
    unavailableTargets: VIDEO_RUNTIME_TARGETS.filter((entry) => entry !== target).map((target) => ({ target, reason: "Fixture unavailable" }))
  });
  const fetchMock = vi.fn(async (url: string | URL | Request) => {
    const payload = payloads.get(String(url));
    if (!payload) throw new Error("No fixture payload for this URL");
    return new Response(new Uint8Array(payload), { status: 200, headers: { "content-length": String(payload.length) } });
  });
  const probe = vi.fn(async () => `ffmpeg version 9.0.2 Copyright fixture\nconfiguration: ${artifact.provenance.configuration}\n`);
  return { artifact, manifest, payloads, fetchMock, probe };
}

async function root(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), "omni-video-runtime-test-"));
  roots.push(directory);
  return directory;
}

afterEach(async () => {
  await Promise.all(roots.splice(0).map((directory) => rm(directory, { recursive: true, force: true })));
});

describe("reviewed automatic video runtime manifest", () => {
  it("ships the published six-target pins and still refuses an explicitly unavailable fixture without fetching", async () => {
    const actual = bundledVideoRuntimeManifest();
    expect(actual.artifacts).toHaveLength(VIDEO_RUNTIME_TARGETS.length);
    expect(actual.unavailableTargets).toEqual([]);
    expect(actual.artifacts.map(artifact => artifact.target)).toEqual([...VIDEO_RUNTIME_TARGETS]);
    expect(actual.artifacts.every(artifact => artifact.version === "9.0.2" && artifact.linkedLibraries[0]?.name === "x264")).toBe(true);
    const manifest = { schemaVersion: 1 as const, component: "FFmpeg" as const, artifacts: [],
      unavailableTargets: VIDEO_RUNTIME_TARGETS.map(target => ({ target, reason: "Deliberately unavailable fixture" })) };
    const fetchMock = vi.fn();
    const directory = await root();
    const provisioner = new VideoRuntimeProvisioner({ manifest, cacheRoot: join(directory, "cache"), platform: "linux", architecture: "x64", environment: { PATH: "" }, fetch: fetchMock });
    await expect(provisioner.prepare()).rejects.toBeInstanceOf(VideoRuntimeUnavailableError);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(await readdir(directory)).toEqual([]);
  });

  it("rejects arbitrary/mutable downloads, commands, missing hashes/sources and nonfree builds", () => {
    const base = fixture();
    const mutate = (change: (value: VideoRuntimeArtifact) => void): void => {
      const artifact = structuredClone(base.artifact);
      change(artifact);
      expect(() => validateVideoRuntimeManifest({ ...base.manifest, artifacts: [artifact] })).toThrow();
    };
    mutate((value) => { value.binary.url = "http://github.com/anything/ffmpeg"; });
    mutate((value) => { value.binary.url = "https://github.com/someone/install/releases/download/v1/ffmpeg"; });
    mutate((value) => { value.binary.url = releaseUrl.replace("video-runtime-fixture-v1", "latest") + "ffmpeg"; });
    mutate((value) => { value.binary.fileName = "../ffmpeg"; });
    mutate((value) => { value.binary.sha256 = ""; });
    mutate((value) => { value.provenance.completeCorrespondingSourceReviewed = false as true; });
    mutate((value) => { value.provenance.configuration = "--enable-nonfree"; });
    mutate((value) => { value.provenance.configuration = "--enable-gpl --enable-libx264"; });
    mutate((value) => { value.provenance.configuration = "--enable-libvpx"; });
    mutate((value) => { (value as unknown as Record<string, unknown>).installCommand = "sh remote.sh"; });
  });

  it("requires corresponding source for every enabled linked library", () => {
    const value = fixture();
    value.artifact.provenance.configuration += " --enable-libvpx";
    value.artifact.linkedLibraries = [{ configureFlag: "--enable-libvpx", name: "libvpx", license: "BSD-3-Clause", source: { ...value.artifact.correspondingSource, fileName: "libvpx-source.tar.xz" } }];
    expect(validateVideoRuntimeManifest({ ...value.manifest, artifacts: [value.artifact] }).artifacts).toHaveLength(1);
    value.artifact.linkedLibraries[0]!.source.sha256 = "not-pinned";
    expect(() => validateVideoRuntimeManifest({ ...value.manifest, artifacts: [value.artifact] })).toThrow(/SHA-256/);
  });
});

describe("automatic runtime cache provisioning without real downloads or execution", () => {
  it("downloads sources/build/notices first, verifies, atomically caches, then reuses without a fetch", async () => {
    const value = fixture();
    const cacheRoot = join(await root(), "cache");
    const progress: VideoRuntimeProgress[] = [];
    const provisioner = new VideoRuntimeProvisioner({ cacheRoot, manifest: value.manifest, platform: "linux", architecture: "x64", environment: { PATH: "" }, fetch: value.fetchMock as typeof fetch, probe: value.probe, readDisk: ampleDisk, onProgress: (entry) => progress.push(entry) });
    const prepared = await provisioner.prepare();
    expect(prepared.state).toBe("ready");
    expect(value.fetchMock.mock.calls.map((call) => String(call[0]))).toEqual(videoRuntimeFiles(value.artifact).map((pin) => pin.url));
    expect(value.probe).toHaveBeenCalledOnce();
    expect(await readFile(join(prepared.sourceDirectory!, "notices.txt"), "utf8")).toContain("fixture notice");
    const receipt = JSON.parse(await readFile(join(prepared.sourceDirectory!, "receipt.json"), "utf8"));
    expect(receipt.artifactSha256).toBe(prepared.artifactSha256);
    expect(progress.at(-1)?.state).toBe("ready");
    value.fetchMock.mockClear();
    expect((await provisioner.prepare()).executablePath).toBe(prepared.executablePath);
    expect(value.fetchMock).not.toHaveBeenCalled();
    expect((await readdir(cacheRoot)).every((name) => !name.startsWith(".install-"))).toBe(true);
  });

  it("preserves a tampered published cache, does not redownload, and rejects its executable", async () => {
    const value = fixture();
    const provisioner = new VideoRuntimeProvisioner({ cacheRoot: join(await root(), "cache"), manifest: value.manifest, platform: "linux", architecture: "x64", environment: { PATH: "" }, fetch: value.fetchMock as typeof fetch, probe: value.probe, readDisk: ampleDisk });
    const installed = await provisioner.prepare();
    value.fetchMock.mockClear();
    await writeFile(installed.executablePath, Buffer.alloc(value.artifact.binary.sizeBytes, 0x78));
    await expect(provisioner.prepare()).rejects.toThrow(/SHA-256/);
    expect(value.fetchMock).not.toHaveBeenCalled();
    expect(await readFile(installed.executablePath)).toEqual(Buffer.alloc(value.artifact.binary.sizeBytes, 0x78));
  });

  it("rejects an incomplete published source payload without re-fetching or deleting the cache", async () => {
    const value = fixture();
    const provisioner = new VideoRuntimeProvisioner({ cacheRoot: join(await root(), "cache"), manifest: value.manifest, platform: "linux", architecture: "x64", environment: { PATH: "" }, fetch: value.fetchMock as typeof fetch, probe: value.probe, readDisk: ampleDisk });
    const installed = await provisioner.prepare();
    value.fetchMock.mockClear();
    await rm(join(installed.sourceDirectory!, value.artifact.correspondingSource.fileName));
    await expect(provisioner.prepare()).rejects.toThrow();
    expect(value.fetchMock).not.toHaveBeenCalled();
    expect(await readFile(installed.executablePath)).toEqual(nativeBytes("linux-x64"));
  });

  it("respects the mandatory disk reserve before starting a fetch", async () => {
    const value = fixture();
    const provisioner = new VideoRuntimeProvisioner({ cacheRoot: join(await root(), "cache"), manifest: value.manifest, platform: "linux", architecture: "x64", environment: { PATH: "" }, fetch: value.fetchMock as typeof fetch, probe: value.probe, readDisk: async () => ({ diskTotalBytes: 30 * 1024 ** 3, diskFreeBytes: MANDATORY_FREE_DISK_BYTES }) });
    await expect(provisioner.prepare()).rejects.toBeInstanceOf(DiskReservePauseError);
    expect(value.fetchMock).not.toHaveBeenCalled();
  });

  it("cleans only its own partial setup on cancellation and never probes it", async () => {
    const value = fixture();
    const cacheRoot = join(await root(), "cache");
    const controller = new AbortController();
    const provisioner = new VideoRuntimeProvisioner({ cacheRoot, manifest: value.manifest, platform: "linux", architecture: "x64", environment: { PATH: "" }, fetch: value.fetchMock as typeof fetch, probe: value.probe, readDisk: ampleDisk, onProgress: (progress) => { if (progress.state === "downloading") controller.abort(new Error("Stop requested")); } });
    await expect(provisioner.prepare(controller.signal)).rejects.toThrow("Stop requested");
    expect(value.fetchMock).toHaveBeenCalledOnce();
    expect(value.probe).not.toHaveBeenCalled();
    expect(await readdir(cacheRoot)).toEqual([]);
  });

  it("rejects wrong hash/size, unsafe redirects and wrong probe configuration without publishing", async () => {
    for (const mode of ["hash", "size", "redirect", "configuration"] as const) {
      const value = fixture();
      const cacheRoot = join(await root(), "cache");
      const fetchMock = mode === "redirect" ? vi.fn(async () => new Response(null, { status: 302, headers: { location: "http://localhost:8000/evil" } })) : mode === "size" ? vi.fn(async () => new Response("oversize", { headers: { "content-length": "9999999" } })) : value.fetchMock;
      if (mode === "hash") value.payloads.set(value.artifact.correspondingSource.url, Buffer.alloc(value.artifact.correspondingSource.sizeBytes, 0));
      const probe = mode === "configuration" ? vi.fn(async () => "ffmpeg version 9.0.2 Copyright fixture\nconfiguration: --enable-nonfree\n") : value.probe;
      const provisioner = new VideoRuntimeProvisioner({ cacheRoot, manifest: value.manifest, platform: "linux", architecture: "x64", environment: { PATH: "" }, fetch: fetchMock as typeof fetch, probe, readDisk: ampleDisk });
      await expect(provisioner.prepare()).rejects.toThrow();
      expect(await readdir(cacheRoot)).toEqual([]);
      if (mode !== "configuration") expect(probe).not.toHaveBeenCalled();
    }
  });

  it("honors explicit host selection (including ordinary symlinks) without claiming Omni verification", async () => {
    const directory = await root();
    const existing = join(directory, "existing-native");
    const selected = join(directory, "selected-native");
    await writeFile(existing, nativeBytes("linux-x64"));
    await chmod(existing, 0o700);
    await symlink(existing, selected);
    const fetchMock = vi.fn();
    const provisioner = new VideoRuntimeProvisioner({ cacheRoot: join(directory, "cache"), platform: "linux", architecture: "x64", environment: { PATH: "", IMAGEIO_FFMPEG_EXE: selected }, fetch: fetchMock });
    const prepared = await provisioner.prepare();
    expect(prepared.state).toBe("external");
    expect(prepared.executablePath).toBe(await realpath(existing));
    expect(prepared.artifactSha256).toBeUndefined();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("accepts native headers for all six targets and rejects script/wrong-architecture headers", async () => {
    const directory = await root();
    for (const target of VIDEO_RUNTIME_TARGETS) {
      const path = join(directory, target);
      await writeFile(path, nativeBytes(target));
      await expect(verifyNativeVideoRuntime(path, target)).resolves.toBeUndefined();
      const other = target.replace(target.endsWith("-x64") ? "-x64" : "-arm64", target.endsWith("-x64") ? "-arm64" : "-x64") as VideoRuntimeTarget;
      await expect(verifyNativeVideoRuntime(path, other)).rejects.toThrow();
      await writeFile(path, Buffer.from("#!/bin/sh\necho not a native binary\n"));
      await expect(verifyNativeVideoRuntime(path, target)).rejects.toThrow();
    }
  });
});
