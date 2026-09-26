import { createHash, randomBytes } from "node:crypto";
import { createReadStream, realpathSync, statSync } from "node:fs";
import {
  open,
  rm,
  rmdir,
  stat
} from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative, resolve } from "node:path";
import type {
  GeneratedArtifact,
  GeneratedArtifactPage,
  ModalityPreview
} from "../shared/types";
import { INLINE_MEDIA_DATA_URL_CHARACTER_LIMIT } from "../shared/mediaTransport";
import {
  ArtifactIndexStore,
  type PersistedGeneratedArtifact
} from "./artifactIndexStore";

export {
  ARTIFACT_INDEX_DATABASE,
  LEGACY_ARTIFACT_INDEX,
  ArtifactIndexStore,
  parseArtifactIndex,
  serializeArtifactIndex,
  type ArtifactIndexPage,
  type PersistedArtifactIndex,
  type PersistedGeneratedArtifact
} from "./artifactIndexStore";

const MEDIA_SCHEME = "omni-media:";
const PREVIEW_TTL_MS = 5 * 60 * 1_000;
const FINAL_LEASE_TTL_MS = 30 * 60 * 1_000;
const MAX_LEASES = 512;

interface MediaLease {
  token: string;
  brainId: string;
  streamKey: string;
  path: string;
  sha256: string;
  mimeType: string;
  size: number;
  device: bigint;
  inode: bigint;
  modifiedMs: number;
  ephemeral: boolean;
  createdAtMs: number;
  expiresAtMs?: number;
  verified?: Promise<void>;
}

export interface AuthorizedMediaArtifact {
  path: string;
  mimeType: string;
  size: number;
  sha256: string;
}

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function inside(path: string, root: string): boolean {
  const child = relative(root, path);
  return child === "" || (
    child !== ".." &&
    !child.startsWith(`..${process.platform === "win32" ? "\\" : "/"}`) &&
    !isAbsolute(child)
  );
}

function safeDataUrl(
  value: unknown,
  expectedMimeType?: string,
  expectedSha256?: string
): string | undefined {
  if (
    typeof value !== "string" ||
    value.length > INLINE_MEDIA_DATA_URL_CHARACTER_LIMIT
  ) return undefined;
  const match = /^data:((?:image|audio|video)\/[a-z0-9.+-]+);base64,([a-z0-9+/]*={0,2})$/i.exec(value);
  if (!match?.[1] || !match[2] || match[2].length % 4 !== 0) return undefined;
  const mimeType = safeMimeType(match[1]);
  if (!mimeType || (expectedMimeType && mimeType !== expectedMimeType)) {
    return undefined;
  }
  if (expectedSha256) {
    const payload = Buffer.from(match[2], "base64");
    if (createHash("sha256").update(payload).digest("hex") !== expectedSha256) {
      return undefined;
    }
  }
  return value;
}

function safeMimeType(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const normalized = value.toLocaleLowerCase();
  const canonical = normalized === "audio/x-wav" || normalized === "audio/wave"
    ? "audio/wav"
    : normalized;
  return /^(?:image|audio|video)\/[a-z0-9.+-]{1,80}$/i.test(canonical)
    ? canonical
    : undefined;
}

function safeSha256(value: unknown): string | undefined {
  return typeof value === "string" && /^[a-f0-9]{64}$/i.test(value)
    ? value.toLocaleLowerCase()
    : undefined;
}

async function sha256File(path: string): Promise<string> {
  return new Promise<string>((resolveHash, rejectHash) => {
    const hash = createHash("sha256");
    const input = createReadStream(path);
    input.on("data", (chunk) => hash.update(chunk));
    input.once("error", rejectHash);
    input.once("end", () => resolveHash(hash.digest("hex")));
  });
}

type OpenFileHandle = Awaited<ReturnType<typeof open>>;

async function readExact(
  handle: OpenFileHandle,
  length: number,
  position: number
): Promise<Buffer | undefined> {
  const buffer = Buffer.alloc(length);
  const { bytesRead } = await handle.read(buffer, 0, length, position);
  return bytesRead === length ? buffer : undefined;
}

/**
 * A RIFF/WAVE magic prefix is not enough for native playback: malformed fmt
 * chunks and unsupported codec tags reach Chromium as an apparently valid
 * resource and surface only as MEDIA_ERR_SRC_NOT_SUPPORTED. Generated WAVs
 * are deliberately limited to the uncompressed codecs Chromium/Electron can
 * decode consistently on every packaged platform.
 */
async function verifiedWaveCodec(
  handle: OpenFileHandle,
  fileSize: number
): Promise<boolean> {
  const prologue = await readExact(handle, 12, 0);
  if (
    !prologue ||
    prologue.subarray(0, 4).toString("ascii") !== "RIFF" ||
    prologue.subarray(8, 12).toString("ascii") !== "WAVE"
  ) return false;
  const riffEnd = prologue.readUInt32LE(4) + 8;
  if (riffEnd < 44 || riffEnd > fileSize) return false;

  let position = 12;
  let blockAlign: number | undefined;
  let dataBytes: number | undefined;
  for (let chunks = 0; chunks < 256 && position + 8 <= riffEnd; chunks += 1) {
    const chunkHeader = await readExact(handle, 8, position);
    if (!chunkHeader) return false;
    const chunkId = chunkHeader.subarray(0, 4).toString("ascii");
    const chunkBytes = chunkHeader.readUInt32LE(4);
    const contentStart = position + 8;
    const contentEnd = contentStart + chunkBytes;
    if (contentEnd > riffEnd) return false;

    if (chunkId === "fmt ") {
      if (chunkBytes < 16) return false;
      const format = await readExact(handle, Math.min(chunkBytes, 40), contentStart);
      if (!format || format.length < 16) return false;
      let codec = format.readUInt16LE(0);
      const channels = format.readUInt16LE(2);
      const sampleRate = format.readUInt32LE(4);
      const byteRate = format.readUInt32LE(8);
      const candidateBlockAlign = format.readUInt16LE(12);
      const bitsPerSample = format.readUInt16LE(14);
      if (codec === 0xfffe) {
        if (
          format.length < 40 ||
          format.readUInt16LE(16) < 22 ||
          format.subarray(28, 40).toString("hex") !==
            "00001000800000aa00389b71"
        ) return false;
        codec = format.readUInt32LE(24);
      }
      const supportedDepth = codec === 1
        ? [8, 16, 24, 32].includes(bitsPerSample)
        : codec === 3 && [32, 64].includes(bitsPerSample);
      const bytesPerSample = bitsPerSample / 8;
      if (
        !supportedDepth ||
        channels < 1 || channels > 8 ||
        sampleRate < 8_000 || sampleRate > 384_000 ||
        !Number.isInteger(bytesPerSample) ||
        candidateBlockAlign !== channels * bytesPerSample ||
        byteRate !== sampleRate * candidateBlockAlign
      ) return false;
      blockAlign = candidateBlockAlign;
    } else if (chunkId === "data") {
      if (chunkBytes < 1) return false;
      dataBytes = chunkBytes;
    }

    if (
      blockAlign !== undefined &&
      dataBytes !== undefined &&
      dataBytes % blockAlign === 0
    ) return true;
    position = contentEnd + (chunkBytes % 2);
  }
  return false;
}

async function verifiedMagic(path: string, mimeType: string): Promise<boolean> {
  const handle = await open(path, "r");
  try {
    if (["audio/wav", "audio/x-wav", "audio/wave"].includes(mimeType)) {
      const file = await handle.stat();
      return await verifiedWaveCodec(handle, file.size);
    }
    const header = Buffer.alloc(16);
    const { bytesRead } = await handle.read(header, 0, header.length, 0);
    const bytes = header.subarray(0, bytesRead);
    if (mimeType === "image/png" || mimeType === "image/apng") {
      return bytes.subarray(0, 8).equals(Buffer.from("89504e470d0a1a0a", "hex"));
    }
    if (mimeType === "video/mp4") {
      return bytes.subarray(4, 8).toString("ascii") === "ftyp";
    }
    return false;
  } finally {
    await handle.close();
  }
}

/**
 * Main-process-only media leases. Filesystem paths are consumed here and are
 * never returned through preload/IPC; the renderer receives an unguessable,
 * content-addressed custom-scheme URL whose bytes are verified on first use.
 */
export class MediaArtifactRegistry {
  private readonly leases = new Map<string, MediaLease>();
  private readonly streamTokens = new Map<string, string>();
  private readonly latestRevisions = new Map<string, number>();
  private readonly indexWrites = new Map<string, Promise<void>>();
  private readonly cleanupTimer: NodeJS.Timeout;

  constructor(
    private readonly brainDirectory: (brainId: string) => string,
    private readonly now: () => number = () => Date.now()
  ) {
    this.cleanupTimer = setInterval(() => this.prune(), 30_000);
    this.cleanupTimer.unref?.();
  }

  private allowedPath(brainId: string, rawPath: string): {
    path: string;
    ephemeral: boolean;
  } {
    if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(brainId)) {
      throw new Error("Invalid media brain id.");
    }
    if (!rawPath || !isAbsolute(rawPath) || rawPath.includes("\0")) {
      throw new Error("Media artifact path must be absolute.");
    }
    const brainRoot = realpathSync(this.brainDirectory(brainId));
    const engineRoot = realpathSync(join(brainRoot, "engine"));
    const actual = realpathSync(resolve(rawPath));
    if (!inside(actual, engineRoot)) {
      throw new Error("Media artifact escaped the selected brain.");
    }
    const finalRoot = resolve(engineRoot, "artifacts");
    const previewRoot = resolve(engineRoot, ".preview-cache");
    const inlineRoot = resolve(engineRoot, ".inline-imagination");
    if (inside(actual, finalRoot)) return { path: actual, ephemeral: false };
    if (inside(actual, previewRoot) || inside(actual, inlineRoot)) {
      return { path: actual, ephemeral: true };
    }
    throw new Error("Media artifact is outside an approved neural output directory.");
  }

  private createLease(input: {
    brainId: string;
    streamKey: string;
    path: string;
    sha256: string;
    mimeType: string;
    ephemeral: boolean;
  }): MediaLease {
    const resolved = this.allowedPath(input.brainId, input.path);
    const mimeType = safeMimeType(input.mimeType);
    if (!mimeType) throw new Error("Media artifact MIME type is invalid.");
    if (resolved.ephemeral !== input.ephemeral) {
      throw new Error("Media artifact lifetime does not match its directory.");
    }
    if (input.ephemeral && !basename(resolved.path).startsWith(input.sha256)) {
      throw new Error("Preview filename is not bound to its content checksum.");
    }
    const file = statSync(resolved.path, { bigint: true });
    if (!file.isFile() || file.size <= 0n || file.size > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error("Media artifact is not a readable finite file.");
    }
    const token = randomBytes(24).toString("hex");
    const createdAtMs = this.now();
    const lease: MediaLease = {
      token,
      brainId: input.brainId,
      streamKey: input.streamKey,
      path: resolved.path,
      sha256: input.sha256,
      mimeType,
      size: Number(file.size),
      device: file.dev,
      inode: file.ino,
      modifiedMs: Number(file.mtimeMs),
      ephemeral: input.ephemeral,
      createdAtMs,
      expiresAtMs: createdAtMs + (
        input.ephemeral ? PREVIEW_TTL_MS : FINAL_LEASE_TTL_MS
      )
    };
    this.leases.set(token, lease);
    this.streamTokens.set(input.streamKey, token);
    this.limitLeases();
    return lease;
  }

  private url(lease: MediaLease): string {
    return `omni-media://artifact/${lease.token}/${lease.sha256}`;
  }

  private revokeToken(token: string): void {
    const lease = this.leases.get(token);
    if (!lease) return;
    this.leases.delete(token);
    if (this.streamTokens.get(lease.streamKey) === token) {
      this.streamTokens.delete(lease.streamKey);
    }
    if (lease.ephemeral) {
      // Remove the capability immediately, then defer physical deletion long
      // enough for an already-open Chromium stream to finish on every OS.
      const path = lease.path;
      setTimeout(() => {
        if ([...this.leases.values()].some((candidate) => candidate.path === path)) {
          return;
        }
        void rm(path, { force: true })
          .then(() => rmdir(dirname(path)))
          .catch(() => undefined);
      }, 30_000).unref?.();
    }
  }

  private revokeStream(streamKey: string): void {
    const token = this.streamTokens.get(streamKey);
    if (token) this.revokeToken(token);
    this.latestRevisions.delete(streamKey);
  }

  private limitLeases(): void {
    if (this.leases.size <= MAX_LEASES) return;
    const candidates = [...this.leases.values()].sort((left, right) =>
      Number(right.ephemeral) - Number(left.ephemeral) ||
      left.createdAtMs - right.createdAtMs
    );
    while (this.leases.size > MAX_LEASES && candidates.length) {
      this.revokeToken(candidates.shift()!.token);
    }
  }

  leasePreview(
    brainId: string,
    streamKey: string,
    preview: ModalityPreview
  ): ModalityPreview | undefined {
    const currentRevision = this.latestRevisions.get(streamKey) ?? -1;
    if (preview.revision <= currentRevision) return undefined;
    const mimeType = safeMimeType(preview.mimeType);
    const sha256 = safeSha256(preview.payloadSha256);
    const artifactPath = preview.artifactPath ?? preview.path;
    const tinyDataUrl = safeDataUrl(preview.dataUrl, mimeType, sha256);
    let mediaUrl: string | undefined;
    if (mimeType && sha256 && artifactPath) {
      const previous = this.streamTokens.get(streamKey);
      let lease: MediaLease;
      try {
        lease = this.createLease({
          brainId,
          streamKey,
          path: artifactPath,
          sha256,
          mimeType,
          ephemeral: true
        });
      } catch {
        return undefined;
      }
      mediaUrl = this.url(lease);
      if (previous && previous !== lease.token) this.revokeToken(previous);
    }
    this.latestRevisions.set(streamKey, preview.revision);
    const sanitized: ModalityPreview = {
      ...preview,
      ...(mimeType ? { mimeType } : {}),
      dataUrl: tinyDataUrl,
      mediaUrl,
      path: undefined,
      artifactPath: undefined
    };
    if (!sanitized.dataUrl && !sanitized.mediaUrl && !sanitized.statusLabel) {
      return undefined;
    }
    return sanitized;
  }

  private async persistArtifact(
    brainId: string,
    artifact: PersistedGeneratedArtifact
  ): Promise<void> {
    const previous = this.indexWrites.get(brainId) ?? Promise.resolve();
    const operation = previous.catch(() => undefined).then(async () => {
      const directory = join(this.brainDirectory(brainId), "engine", "artifacts");
      const store = await ArtifactIndexStore.open(directory, brainId);
      try {
        store.append([artifact]);
      } finally {
        store.close();
      }
    });
    this.indexWrites.set(brainId, operation);
    try {
      await operation;
    } finally {
      if (this.indexWrites.get(brainId) === operation) {
        this.indexWrites.delete(brainId);
      }
    }
  }

  async completeJob(
    brainId: string,
    jobId: string,
    outputValue: unknown
  ): Promise<unknown> {
    const output = record(outputValue);
    if (!output) {
      this.revokeStream(`job:${brainId}:${jobId}`);
      return outputValue;
    }
    const mimeType = safeMimeType(output.mimeType);
    const sha256 = safeSha256(output.artifactSha256);
    const rawPath = typeof output.path === "string"
      ? output.path
      : typeof output.artifactPath === "string"
        ? output.artifactPath
        : undefined;
    let mediaUrl: string | undefined;
    if (mimeType && sha256 && rawPath) {
      const lease = this.createLease({
        brainId,
        streamKey: `final:${brainId}:${jobId}`,
        path: rawPath,
        sha256,
        mimeType,
        ephemeral: false
      });
      mediaUrl = this.url(lease);
      await this.verify(lease);
      const modality = output.modality;
      if (!["image", "audio", "video"].includes(String(modality))) {
        throw new Error("Generated artifact modality is invalid.");
      }
      await this.persistArtifact(brainId, {
        id: createHash("sha256").update(`${brainId}:${jobId}`).digest("hex"),
        modality: modality as PersistedGeneratedArtifact["modality"],
        mimeType,
        sha256,
        bytes: lease.size,
        relativePath: `artifacts/${basename(lease.path)}`,
        createdAt: new Date(this.now()).toISOString(),
        ...(typeof output.seed === "number" && Number.isSafeInteger(output.seed)
          ? { seed: output.seed }
          : {}),
        ...(typeof output.initialization === "string"
          ? { initialization: output.initialization.slice(0, 240) }
          : {}),
        ...(typeof output.qualityNote === "string"
          ? { qualityNote: output.qualityNote.slice(0, 1_000) }
          : {})
      });
    }
    this.revokeStream(`job:${brainId}:${jobId}`);
    const neuralActionId = typeof output.neuralActionId === "string"
      ? output.neuralActionId
      : "";
    if (neuralActionId) {
      this.revokeStream(`action:${brainId}:${neuralActionId}`);
    }
    const { path: _path, artifactPath: _artifactPath, dataUrl, ...rest } = output;
    return {
      ...rest,
      ...(mimeType ? { mimeType } : {}),
      ...(safeDataUrl(dataUrl, mimeType, sha256) ? { dataUrl } : {}),
      ...(mediaUrl ? { mediaUrl } : {})
    };
  }

  async listArtifacts(
    brainId: string,
    cursor?: string,
    limitValue = 48
  ): Promise<GeneratedArtifactPage> {
    const directory = join(this.brainDirectory(brainId), "engine", "artifacts");
    const store = await ArtifactIndexStore.open(directory, brainId);
    let page;
    try {
      page = store.page(cursor, limitValue);
    } finally {
      store.close();
    }
    const root = this.brainDirectory(brainId);
    const output: GeneratedArtifact[] = [];
    for (const artifact of page.artifacts) {
      const path = join(root, "engine", ...artifact.relativePath.split("/"));
      try {
        const lease = this.createLease({
          brainId,
          streamKey: `history:${brainId}:${artifact.id}`,
          path,
          sha256: artifact.sha256,
          mimeType: artifact.mimeType,
          ephemeral: false
        });
        await this.verify(lease);
        output.push({
          id: artifact.id,
          brainId,
          modality: artifact.modality,
          mimeType: artifact.mimeType,
          sha256: artifact.sha256,
          bytes: artifact.bytes,
          createdAt: artifact.createdAt,
          seed: artifact.seed,
          initialization: artifact.initialization,
          qualityNote: artifact.qualityNote,
          mediaUrl: this.url(lease),
          available: true
        });
      } catch (error) {
        output.push({
          id: artifact.id,
          brainId,
          modality: artifact.modality,
          mimeType: artifact.mimeType,
          sha256: artifact.sha256,
          bytes: artifact.bytes,
          createdAt: artifact.createdAt,
          seed: artifact.seed,
          initialization: artifact.initialization,
          qualityNote: artifact.qualityNote,
          available: false,
          unavailableReason: (error as NodeJS.ErrnoException).code === "ENOENT"
            ? "missing"
            : "integrity-failed"
        });
      }
    }
    return {
      brainId,
      artifacts: output,
      totalArtifacts: page.totalArtifacts,
      ...(page.nextCursor ? { nextCursor: page.nextCursor } : {})
    };
  }

  cancelJob(brainId: string, jobId: string): void {
    this.revokeStream(`job:${brainId}:${jobId}`);
  }

  private async verify(lease: MediaLease): Promise<void> {
    if (lease.verified) return lease.verified;
    lease.verified = (async () => {
      const current = await stat(lease.path, { bigint: true });
      if (
        !current.isFile() ||
        Number(current.size) !== lease.size ||
        current.dev !== lease.device ||
        current.ino !== lease.inode ||
        Number(current.mtimeMs) !== lease.modifiedMs
      ) {
        throw new Error("Media artifact changed after its lease was issued.");
      }
      if (!await verifiedMagic(lease.path, lease.mimeType)) {
        throw new Error("Media artifact signature does not match its MIME type.");
      }
      if (await sha256File(lease.path) !== lease.sha256) {
        throw new Error("Media artifact checksum does not match its lease.");
      }
    })().catch((error) => {
      this.revokeToken(lease.token);
      throw error;
    });
    return lease.verified;
  }

  async authorize(rawUrl: string): Promise<AuthorizedMediaArtifact | undefined> {
    let url: URL;
    try {
      url = new URL(rawUrl);
    } catch {
      return undefined;
    }
    if (
      url.protocol !== MEDIA_SCHEME ||
      url.hostname !== "artifact" ||
      url.username ||
      url.password ||
      url.port ||
      url.search ||
      url.hash
    ) {
      return undefined;
    }
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts.length !== 2) return undefined;
    const [token, sha256] = parts;
    const lease = token ? this.leases.get(token) : undefined;
    if (!lease || sha256 !== lease.sha256) return undefined;
    if (lease.expiresAtMs !== undefined && lease.expiresAtMs <= this.now()) {
      this.revokeToken(lease.token);
      return undefined;
    }
    await this.verify(lease);
    return {
      path: lease.path,
      mimeType: lease.mimeType,
      size: lease.size,
      sha256: lease.sha256
    };
  }

  prune(): void {
    const now = this.now();
    for (const lease of this.leases.values()) {
      if (lease.expiresAtMs !== undefined && lease.expiresAtMs <= now) {
        this.revokeToken(lease.token);
      }
    }
  }

  dispose(): void {
    clearInterval(this.cleanupTimer);
    for (const lease of [...this.leases.values()]) this.revokeToken(lease.token);
    this.leases.clear();
    this.streamTokens.clear();
    this.latestRevisions.clear();
  }
}
