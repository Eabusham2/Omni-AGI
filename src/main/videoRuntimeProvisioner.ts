import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { constants, createReadStream } from "node:fs";
import { access, chmod, lstat, mkdir, mkdtemp, open, readFile, realpath, rename, rm, writeFile } from "node:fs/promises";
import { delimiter, join, resolve } from "node:path";
import { requireDiskWrite, type DiskSpaceReader } from "./diskSpace";
import {
  bundledVideoRuntimeManifest,
  validateVideoRuntimeDownloadUrl,
  validateVideoRuntimeManifest,
  videoRuntimeArtifactHash,
  videoRuntimeFiles,
  type VideoRuntimeArtifact,
  type VideoRuntimeFilePin,
  type VideoRuntimeManifest,
  type VideoRuntimeTarget
} from "./videoRuntimeManifest";

export interface VideoRuntimeProgress {
  state: "checking" | "downloading" | "verifying" | "ready" | "external" | "unavailable";
  target: string;
  message: string;
  completedBytes?: number;
  totalBytes?: number;
  fileName?: string;
}

export interface PreparedVideoRuntime {
  state: "ready" | "external";
  executablePath: string;
  sourceDirectory?: string;
  artifactId?: string;
  artifactSha256?: string;
  binarySha256?: string;
  binarySizeBytes?: number;
  target?: VideoRuntimeTarget;
}

export class VideoRuntimeUnavailableError extends Error {
  readonly target: string;
  constructor(target: string, reason: string) {
    super(`Automatic video runtime is unavailable for ${target}: ${reason}`);
    this.name = "VideoRuntimeUnavailableError";
    this.target = target;
  }
}

export interface VideoRuntimeProvisionerOptions {
  cacheRoot: string;
  manifest?: VideoRuntimeManifest;
  platform?: NodeJS.Platform;
  architecture?: string;
  environment?: NodeJS.ProcessEnv;
  readDisk?: DiskSpaceReader;
  fetch?: typeof globalThis.fetch;
  /** Only tests override this. Never read probe commands from downloaded metadata. */
  probe?: (path: string, signal?: AbortSignal) => Promise<string>;
  onProgress?: (progress: VideoRuntimeProgress) => void;
}

const REDIRECT_HOSTS = new Set(["github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"]);
const MAX_REPORT_BYTES = 256 * 1024;
const DISK_CHECK_INTERVAL = 8 * 1024 * 1024;
const activeInstalls = new Map<string, Promise<PreparedVideoRuntime>>();

function abortIfNeeded(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw signal.reason instanceof Error ? signal.reason : new Error("Video runtime setup was cancelled.");
  }
}

async function regularDirectory(path: string): Promise<void> {
  const metadata = await lstat(path);
  if (!metadata.isDirectory() || metadata.isSymbolicLink()) throw new Error(`Video runtime cache directory is not a regular directory: ${path}`);
}

async function hashPinnedFile(path: string, pin: VideoRuntimeFilePin, signal?: AbortSignal): Promise<void> {
  abortIfNeeded(signal);
  const metadata = await lstat(path);
  if (!metadata.isFile() || metadata.isSymbolicLink() || metadata.size !== pin.sizeBytes) {
    throw new Error(`Video runtime ${pin.fileName} is missing, redirected or has the wrong byte size.`);
  }
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path, { highWaterMark: 1024 * 1024, signal })) {
    abortIfNeeded(signal);
    hash.update(chunk as Buffer);
  }
  if (hash.digest("hex") !== pin.sha256) throw new Error(`Video runtime ${pin.fileName} SHA-256 does not match the reviewed artifact.`);
}

/** Reject shell scripts and wrong-architecture executables before any process starts. */
export async function verifyNativeVideoRuntime(path: string, target: VideoRuntimeTarget): Promise<void> {
  const file = await open(path, "r");
  try {
    const header = Buffer.alloc(64);
    const { bytesRead } = await file.read(header, 0, header.length, 0);
    if (bytesRead < 20) throw new Error("FFmpeg native executable header is truncated.");
    if (target.startsWith("linux-")) {
      const machine = target.endsWith("-x64") ? 62 : 183;
      if (!header.subarray(0, 4).equals(Buffer.from([0x7f, 0x45, 0x4c, 0x46])) ||
          header[4] !== 2 || header[5] !== 1 || header.readUInt16LE(18) !== machine) {
        throw new Error("FFmpeg executable is not an ELF64 binary for the selected architecture.");
      }
    } else if (target.startsWith("darwin-")) {
      const machine = target.endsWith("-x64") ? 0x01000007 : 0x0100000c;
      if (header.readUInt32LE(0) !== 0xfeedfacf || header.readUInt32LE(4) !== machine) {
        throw new Error("FFmpeg executable is not a native Mach-O 64-bit binary for the selected architecture.");
      }
    } else {
      if (bytesRead < 64 || header[0] !== 0x4d || header[1] !== 0x5a) throw new Error("FFmpeg executable is not a Windows PE binary.");
      const peOffset = header.readUInt32LE(60);
      if (peOffset < 64 || peOffset > 1024 * 1024) throw new Error("FFmpeg PE header offset is invalid.");
      const pe = Buffer.alloc(6);
      const read = await file.read(pe, 0, pe.length, peOffset);
      const machine = target.endsWith("-x64") ? 0x8664 : 0xaa64;
      if (read.bytesRead !== pe.length || pe.readUInt32LE(0) !== 0x00004550 || pe.readUInt16LE(4) !== machine) {
        throw new Error("FFmpeg executable is not a PE binary for the selected architecture.");
      }
    }
  } finally {
    await file.close();
  }
}

async function defaultProbe(path: string, signal?: AbortSignal): Promise<string> {
  abortIfNeeded(signal);
  return new Promise((resolveProbe, reject) => {
    const child = spawn(path, ["-version"], { shell: false, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
    const chunks: Buffer[] = [];
    let size = 0;
    let failure: Error | undefined;
    const stop = (error: Error): void => {
      failure ??= error;
      child.kill("SIGKILL");
    };
    const collect = (value: Buffer): void => {
      size += value.byteLength;
      if (size > MAX_REPORT_BYTES) stop(new Error("FFmpeg runtime version report exceeds the safe bound."));
      else chunks.push(value);
    };
    const onAbort = (): void => stop(new Error("Video runtime verification was cancelled."));
    const timeout = setTimeout(() => stop(new Error("FFmpeg runtime verification timed out.")), 10_000);
    signal?.addEventListener("abort", onAbort, { once: true });
    if (signal?.aborted) onAbort();
    child.stdout.on("data", collect);
    child.stderr.on("data", collect);
    child.once("error", (error) => { failure ??= error; });
    child.once("close", (code) => {
      clearTimeout(timeout);
      signal?.removeEventListener("abort", onAbort);
      if (failure) reject(failure);
      else if (code !== 0) reject(new Error(`FFmpeg runtime verification failed with exit code ${String(code)}.`));
      else resolveProbe(Buffer.concat(chunks).toString("utf8"));
    });
  });
}

function verifyProbeReport(report: string, artifact: VideoRuntimeArtifact): void {
  if (Buffer.byteLength(report) > MAX_REPORT_BYTES || !report.startsWith(`ffmpeg version ${artifact.version} `)) {
    throw new Error("FFmpeg version does not match the reviewed runtime.");
  }
  const configuration = report.split(/\r?\n/).find((line) => line.startsWith("configuration: "));
  if (configuration?.slice("configuration: ".length) !== artifact.provenance.configuration) {
    throw new Error("FFmpeg build configuration does not match the reviewed runtime.");
  }
}

async function externalExecutable(environment: NodeJS.ProcessEnv, platform: NodeJS.Platform): Promise<string | undefined> {
  const explicit = environment.IMAGEIO_FFMPEG_EXE?.trim();
  const fileName = platform === "win32" ? "ffmpeg.exe" : "ffmpeg";
  const candidates = explicit ? [resolve(explicit)] : (environment.PATH ?? "")
    .split(delimiter).filter(Boolean).map((directory) => join(directory, fileName));
  for (const path of candidates) {
    try {
      const resolved = await realpath(path);
      const metadata = await lstat(resolved);
      if (!metadata.isFile()) continue;
      await access(resolved, platform === "win32" ? constants.R_OK : constants.X_OK);
      return resolved;
    } catch {
      // Explicit invalid selections are reported, never silently replaced by an automatic runtime.
    }
  }
  if (explicit) throw new Error("The selected IMAGEIO_FFMPEG_EXE is not an accessible executable file.");
  return undefined;
}

export class VideoRuntimeProvisioner {
  private readonly options: VideoRuntimeProvisionerOptions;
  private readonly manifest: VideoRuntimeManifest;
  constructor(options: VideoRuntimeProvisionerOptions) {
    this.options = { ...options, cacheRoot: resolve(options.cacheRoot) };
    this.manifest = validateVideoRuntimeManifest(options.manifest ?? bundledVideoRuntimeManifest());
  }

  private progress(progress: VideoRuntimeProgress): void {
    this.options.onProgress?.(progress);
  }

  async prepare(signal?: AbortSignal): Promise<PreparedVideoRuntime> {
    abortIfNeeded(signal);
    const platform = this.options.platform ?? process.platform;
    const runtimeTarget = `${platform}-${this.options.architecture ?? process.arch}`;
    this.progress({ state: "checking", target: runtimeTarget, message: "Checking the selected and reviewed video runtime." });
    const external = await externalExecutable(this.options.environment ?? process.env, platform);
    abortIfNeeded(signal);
    if (external) {
      this.progress({ state: "external", target: runtimeTarget, message: "Using the user-selected or existing host FFmpeg; this is not a verified Omni runtime." });
      return { state: "external", executablePath: external };
    }
    const artifact = this.manifest.artifacts.find((entry) => entry.target === runtimeTarget);
    if (!artifact) {
      const reason = this.manifest.unavailableTargets.find((entry) => entry.target === runtimeTarget)?.reason ?? "This platform/architecture is not supported.";
      this.progress({ state: "unavailable", target: runtimeTarget, message: reason });
      throw new VideoRuntimeUnavailableError(runtimeTarget, reason);
    }
    const artifactHash = videoRuntimeArtifactHash(artifact);
    const finalDirectory = join(this.options.cacheRoot, `${artifact.id}-${artifactHash}`);
    const existingInstall = activeInstalls.get(finalDirectory);
    if (existingInstall) {
      // The install owner's Stop cancels its download. A different caller may
      // stop waiting without aborting someone else's explicitly owned work.
      await this.waitForInstall(existingInstall, signal);
      return this.verifyCache(finalDirectory, artifact, signal);
    }
    const installing = this.install(finalDirectory, artifact, signal);
    activeInstalls.set(finalDirectory, installing);
    try {
      return await installing;
    } finally {
      if (activeInstalls.get(finalDirectory) === installing) activeInstalls.delete(finalDirectory);
    }
  }

  private async waitForInstall(value: Promise<PreparedVideoRuntime>, signal?: AbortSignal): Promise<void> {
    if (!signal) { await value; return; }
    abortIfNeeded(signal);
    await new Promise<void>((resolveWait, reject) => {
      const onAbort = (): void => reject(signal.reason instanceof Error ? signal.reason : new Error("Video runtime setup wait was cancelled."));
      signal.addEventListener("abort", onAbort, { once: true });
      value.then(() => { signal.removeEventListener("abort", onAbort); resolveWait(); }, (error: unknown) => { signal.removeEventListener("abort", onAbort); reject(error); });
    });
  }

  private async verifyCache(directory: string, artifact: VideoRuntimeArtifact, signal?: AbortSignal): Promise<PreparedVideoRuntime> {
    await regularDirectory(this.options.cacheRoot);
    await regularDirectory(directory);
    for (const pin of videoRuntimeFiles(artifact)) await hashPinnedFile(join(directory, pin.fileName), pin, signal);
    const receiptPath = join(directory, "receipt.json");
    const receiptMetadata = await lstat(receiptPath);
    if (!receiptMetadata.isFile() || receiptMetadata.isSymbolicLink() || receiptMetadata.size > 512 * 1024) {
      throw new Error("Video runtime install receipt is missing or unsafe.");
    }
    const receipt = JSON.parse(await readFile(receiptPath, "utf8")) as { schemaVersion?: unknown; artifactSha256?: unknown; artifact?: unknown };
    if (receipt?.schemaVersion !== 1 || receipt.artifactSha256 !== videoRuntimeArtifactHash(artifact) ||
        JSON.stringify(receipt.artifact) !== JSON.stringify(artifact)) {
      throw new Error("Video runtime install receipt does not match the reviewed artifact.");
    }
    const executablePath = await realpath(join(directory, artifact.binary.fileName));
    await verifyNativeVideoRuntime(executablePath, artifact.target);
    await access(executablePath, artifact.target.startsWith("win32-") ? constants.R_OK : constants.X_OK);
    abortIfNeeded(signal);
    this.progress({ state: "ready", target: artifact.target, message: "The pinned video executable, matching sources and notices are verified." });
    return {
      state: "ready", executablePath, sourceDirectory: directory,
      artifactId: artifact.id, artifactSha256: videoRuntimeArtifactHash(artifact),
      binarySha256: artifact.binary.sha256, binarySizeBytes: artifact.binary.sizeBytes,
      target: artifact.target
    };
  }

  private async install(finalDirectory: string, artifact: VideoRuntimeArtifact, signal?: AbortSignal): Promise<PreparedVideoRuntime> {
    abortIfNeeded(signal);
    await mkdir(this.options.cacheRoot, { recursive: true, mode: 0o700 });
    await regularDirectory(this.options.cacheRoot);
    let cached = false;
    try {
      await lstat(finalDirectory);
      cached = true;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    // Never erase or redownload a tampered/incomplete published cache as a repair shortcut.
    if (cached) return this.verifyCache(finalDirectory, artifact, signal);
    const files = videoRuntimeFiles(artifact);
    const receipt = JSON.stringify({
      schemaVersion: 1, artifactSha256: videoRuntimeArtifactHash(artifact), artifact,
      installedAt: new Date().toISOString()
    }, null, 2);
    const receiptBytes = Buffer.byteLength(receipt);
    if (receiptBytes > 512 * 1024) throw new Error("Video runtime receipt exceeds its safe bound.");
    const totalBytes = files.reduce((sum, file) => sum + file.sizeBytes, 0) + receiptBytes;
    await requireDiskWrite(this.options.cacheRoot, { operationWriteBytes: totalBytes }, { read: this.options.readDisk, platform: this.options.platform });
    abortIfNeeded(signal);
    const temporary = await mkdtemp(join(this.options.cacheRoot, ".install-"));
    let completedBytes = 0;
    try {
      for (const pin of files) {
        await this.download(pin, join(temporary, pin.fileName), artifact, completedBytes, totalBytes, signal);
        completedBytes += pin.sizeBytes;
      }
      this.progress({ state: "verifying", target: artifact.target, completedBytes, totalBytes, message: "Verifying exact native binary, build configuration, sources and license notices." });
      for (const pin of files) await hashPinnedFile(join(temporary, pin.fileName), pin, signal);
      const executablePath = join(temporary, artifact.binary.fileName);
      await verifyNativeVideoRuntime(executablePath, artifact.target);
      await chmod(executablePath, 0o700);
      verifyProbeReport(await (this.options.probe ?? defaultProbe)(executablePath, signal), artifact);
      abortIfNeeded(signal);
      await requireDiskWrite(this.options.cacheRoot, { operationWriteBytes: receiptBytes }, { read: this.options.readDisk, platform: this.options.platform });
      await writeFile(join(temporary, "receipt.json"), receipt, { flag: "wx", mode: 0o600 });
      abortIfNeeded(signal);
      try {
        await rename(temporary, finalDirectory);
      } catch (error) {
        if (!["EEXIST", "ENOTEMPTY"].includes(String((error as NodeJS.ErrnoException).code))) throw error;
        // Another app process may have atomically published the same payload.
      }
      return await this.verifyCache(finalDirectory, artifact, signal);
    } finally {
      // The exact mkdtemp directory belongs to this attempt. No user/existing
      // published runtime is deleted, even on hash failure or cancellation.
      await rm(temporary, { recursive: true, force: true });
    }
  }

  private async download(
    pin: VideoRuntimeFilePin, path: string, artifact: VideoRuntimeArtifact,
    completedBytes: number, totalBytes: number, signal?: AbortSignal
  ): Promise<void> {
    abortIfNeeded(signal);
    let url = validateVideoRuntimeDownloadUrl(pin.url);
    let response: Response | undefined;
    for (let redirects = 0; redirects <= 4; redirects += 1) {
      abortIfNeeded(signal);
      const fetched = await (this.options.fetch ?? globalThis.fetch)(url.toString(), { redirect: "manual", signal });
      if ([301, 302, 303, 307, 308].includes(fetched.status)) {
        const location = fetched.headers.get("location");
        await fetched.body?.cancel();
        if (!location || redirects === 4) throw new Error("Video runtime redirect chain is missing or too long.");
        const redirect = new URL(location, url);
        if (redirect.protocol !== "https:" || redirect.username || redirect.password || redirect.port || !REDIRECT_HOSTS.has(redirect.hostname)) {
          throw new Error("Video runtime download redirected outside the trusted HTTPS asset hosts.");
        }
        url = redirect;
      } else {
        response = fetched;
        break;
      }
    }
    if (!response?.ok || !response.body) throw new Error(`Video runtime ${pin.fileName} download failed with HTTP ${String(response?.status)}.`);
    const contentLength = response.headers.get("content-length");
    if (contentLength && (!/^\d+$/.test(contentLength) || Number(contentLength) !== pin.sizeBytes)) {
      await response.body.cancel();
      throw new Error(`Video runtime ${pin.fileName} content length does not match its pin.`);
    }
    const reader = response.body.getReader();
    const file = await open(path, "wx", 0o600);
    const hash = createHash("sha256");
    let downloadedBytes = 0;
    let lastDiskCheck = -DISK_CHECK_INTERVAL;
    const onAbort = (): void => { void reader.cancel().catch(() => undefined); };
    signal?.addEventListener("abort", onAbort, { once: true });
    try {
      while (true) {
        abortIfNeeded(signal);
        const chunk = await reader.read();
        abortIfNeeded(signal);
        if (chunk.done) break;
        if (downloadedBytes + chunk.value.byteLength > pin.sizeBytes) throw new Error(`Video runtime ${pin.fileName} download exceeds its exact pinned size.`);
        if (downloadedBytes - lastDiskCheck >= DISK_CHECK_INTERVAL) {
          await requireDiskWrite(this.options.cacheRoot, { operationWriteBytes: totalBytes - completedBytes - downloadedBytes }, { read: this.options.readDisk, platform: this.options.platform });
          lastDiskCheck = downloadedBytes;
        }
        // FileHandle.write may write a prefix. Persist every byte before advancing.
        let offset = 0;
        while (offset < chunk.value.byteLength) {
          abortIfNeeded(signal);
          const written = await file.write(chunk.value, offset, chunk.value.byteLength - offset);
          if (written.bytesWritten <= 0) throw new Error("Video runtime cache write made no progress.");
          offset += written.bytesWritten;
        }
        hash.update(chunk.value);
        downloadedBytes += chunk.value.byteLength;
        this.progress({ state: "downloading", target: artifact.target, fileName: pin.fileName,
          completedBytes: completedBytes + downloadedBytes, totalBytes,
          message: `Video setup: ${pin.fileName}, ${(completedBytes + downloadedBytes).toLocaleString()}/${totalBytes.toLocaleString()} bytes; sources/notices before executable.` });
      }
      if (downloadedBytes !== pin.sizeBytes || hash.digest("hex") !== pin.sha256) {
        throw new Error(`Video runtime ${pin.fileName} failed exact size/SHA-256 verification.`);
      }
      await file.sync();
    } finally {
      signal?.removeEventListener("abort", onAbort);
      await reader.cancel().catch(() => undefined);
      await file.close();
    }
  }
}
