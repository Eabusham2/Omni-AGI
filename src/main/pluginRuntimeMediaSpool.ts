import { createHash } from "node:crypto";
import { constants, createReadStream } from "node:fs";
import { lstat, mkdir, open, rename, rm, stat, type FileHandle } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { normalizeSourceMediaType } from "../shared/sourceMediaType";
import { extractStreamingZip, writeStreamingZip } from "./streamingZip";

const SHA256 = /^[a-f0-9]{64}$/;
const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const MAGIC_PREFIX_BYTES = 64;
const ASSET_METADATA_BYTES = 2_048;
const ARCHIVE_WORKING_BYTES = 512 * 1024;

export type PluginRuntimeMediaKind = "image" | "audio" | "video";

export interface PluginRuntimeMediaIdentity {
  pluginId: string;
  packageSha256: string;
  formatId: string;
  moduleId: string;
  moduleSha256: string;
  transactionKey: string;
  sourceSha256: string;
  sourceBytes: number;
}

export interface PluginRuntimeMediaAssetDescriptor {
  ordinal: number;
  assetId: string;
  kind: PluginRuntimeMediaKind;
  mimeType: string;
  memberName: string;
  bytes: number;
  sha256: string;
}

export interface PluginRuntimeMediaBundleProvenance extends PluginRuntimeMediaIdentity {
  format: "omni-runtime-parser-media-bundle";
  version: 1;
  assets: PluginRuntimeMediaAssetDescriptor[];
  assetManifestSha256: string;
  modalityCounts: Record<PluginRuntimeMediaKind, number>;
  bundleBytes: number;
  bundleSha256: string;
  complete: true;
}

function canonical(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  return `{${Object.entries(value as Record<string, unknown>).sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
    .map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(",")}}`;
}

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

export interface PluginRuntimeMediaSpoolResult {
  state: "ready";
  bundlePath: string;
  provenance: PluginRuntimeMediaBundleProvenance;
  cleanup(): Promise<void>;
}

export interface PluginRuntimeMediaEmptyResult {
  state: "no-assets";
  assets: [];
  assetManifestSha256: string;
  modalityCounts: Record<PluginRuntimeMediaKind, 0>;
  reason: "no-assets-no-modality-updates";
  cleanup(): Promise<void>;
}

export interface PluginRuntimeMediaSpoolOptions {
  workspaceRoot: string;
  identity: PluginRuntimeMediaIdentity;
  allowedMediaTypes: readonly string[];
  maximumChunkBytes: number;
  signal?: AbortSignal;
  admitMemory(bytes: number, signal?: AbortSignal): Promise<{ release(): void | Promise<void> }>;
  checkDisk(directory: string, operationWriteBytes: number, signal?: AbortSignal): Promise<void>;
}

interface SupportedMediaType {
  kind: PluginRuntimeMediaKind;
  extension: string;
  validMagic(prefix: Buffer): boolean;
}

interface OwnedAsset extends PluginRuntimeMediaAssetDescriptor {
  path: string;
  metadataLease: { release(): void | Promise<void> };
}

interface ActiveAsset {
  ordinal: number;
  assetId: string;
  kind: PluginRuntimeMediaKind;
  mimeType: string;
  memberName: string;
  temporaryPath: string;
  finalPath: string;
  handle: FileHandle;
  hash: ReturnType<typeof createHash>;
  bytes: number;
  prefix: Buffer;
  metadataLease: { release(): void | Promise<void> };
}

function starts(prefix: Buffer, bytes: readonly number[]): boolean {
  return prefix.length >= bytes.length && bytes.every((value, index) => prefix[index] === value);
}

function ascii(prefix: Buffer, offset: number, value: string): boolean {
  return prefix.length >= offset + value.length && prefix.subarray(offset, offset + value.length).toString("ascii") === value;
}

const SUPPORTED_MEDIA = new Map<string, SupportedMediaType>([
  ["image/png", { kind: "image", extension: "png", validMagic: prefix => starts(prefix, [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]) }],
  ["image/jpeg", { kind: "image", extension: "jpg", validMagic: prefix => starts(prefix, [0xff, 0xd8, 0xff]) }],
  ["image/gif", { kind: "image", extension: "gif", validMagic: prefix => ascii(prefix, 0, "GIF87a") || ascii(prefix, 0, "GIF89a") }],
  ["image/webp", { kind: "image", extension: "webp", validMagic: prefix => ascii(prefix, 0, "RIFF") && ascii(prefix, 8, "WEBP") }],
  ["audio/wav", { kind: "audio", extension: "wav", validMagic: prefix => ascii(prefix, 0, "RIFF") && ascii(prefix, 8, "WAVE") }],
  ["audio/x-wav", { kind: "audio", extension: "wav", validMagic: prefix => ascii(prefix, 0, "RIFF") && ascii(prefix, 8, "WAVE") }],
  ["audio/flac", { kind: "audio", extension: "flac", validMagic: prefix => ascii(prefix, 0, "fLaC") }],
  ["audio/mpeg", { kind: "audio", extension: "mp3", validMagic: prefix => ascii(prefix, 0, "ID3") || (prefix.length >= 2 && prefix[0] === 0xff && (prefix[1]! & 0xe0) === 0xe0) }],
  ["video/mp4", { kind: "video", extension: "mp4", validMagic: prefix => ascii(prefix, 4, "ftyp") }],
  ["video/quicktime", { kind: "video", extension: "mov", validMagic: prefix => ascii(prefix, 4, "ftyp") }],
  ["video/webm", { kind: "video", extension: "webm", validMagic: prefix => starts(prefix, [0x1a, 0x45, 0xdf, 0xa3]) }]
]);

function validIdentity(identity: PluginRuntimeMediaIdentity): boolean {
  return SAFE_ID.test(identity.pluginId) && SAFE_ID.test(identity.formatId) && SAFE_ID.test(identity.moduleId) &&
    SHA256.test(identity.packageSha256) && SHA256.test(identity.moduleSha256) && SHA256.test(identity.transactionKey) &&
    SHA256.test(identity.sourceSha256) && Number.isSafeInteger(identity.sourceBytes) && identity.sourceBytes >= 0;
}

function decodedLength(value: string): number {
  if (!value || value.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(value)) {
    throw new Error("Runtime media parser returned malformed base64 bytes.");
  }
  const padding = value.endsWith("==") ? 2 : value.endsWith("=") ? 1 : 0;
  const length = value.length / 4 * 3 - padding;
  if (!Number.isSafeInteger(length) || length <= 0) throw new Error("Runtime media parser returned an empty or unsafe byte page.");
  return length;
}

async function writeAll(handle: FileHandle, bytes: Buffer, position: number): Promise<void> {
  let offset = 0;
  while (offset < bytes.length) {
    const result = await handle.write(bytes, offset, bytes.length - offset, position + offset);
    if (result.bytesWritten < 1) throw new Error("Runtime media asset write stopped before the full admitted page was stored.");
    offset += result.bytesWritten;
  }
}

async function regularFileSize(path: string): Promise<number> {
  const value = await lstat(path);
  if (!value.isFile() || value.isSymbolicLink() || !Number.isSafeInteger(value.size) || value.size < 0) {
    throw new Error("Runtime media custody is not an exact regular file.");
  }
  return value.size;
}

async function fileSha256(path: string, signal?: AbortSignal): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path, { ...(signal ? { signal } : {}) })) {
    signal?.throwIfAborted();
    hash.update(chunk as Buffer);
  }
  signal?.throwIfAborted();
  return hash.digest("hex");
}

export async function verifyPluginRuntimeMediaBundle(
  bundlePath: string,
  provenance: PluginRuntimeMediaBundleProvenance,
  controls: Pick<PluginRuntimeMediaSpoolOptions, "signal" | "admitMemory" | "checkDisk">
): Promise<void> {
  const provenanceKeys = ["assetManifestSha256", "assets", "bundleBytes", "bundleSha256", "complete", "format", "formatId", "modalityCounts",
    "moduleId", "moduleSha256", "packageSha256", "pluginId", "sourceBytes", "sourceSha256", "transactionKey", "version"];
  if (Object.keys(provenance).sort().join("\0") !== provenanceKeys.sort().join("\0") || !validIdentity(provenance) ||
      provenance.format !== "omni-runtime-parser-media-bundle" || provenance.version !== 1 ||
      !SHA256.test(provenance.assetManifestSha256) || !SHA256.test(provenance.bundleSha256) ||
      !Number.isSafeInteger(provenance.bundleBytes) || provenance.bundleBytes < 1 || !Array.isArray(provenance.assets) || !provenance.assets.length ||
      Object.keys(provenance.modalityCounts).sort().join("\0") !== ["audio", "image", "video"].join("\0") ||
      ![provenance.modalityCounts.image, provenance.modalityCounts.audio, provenance.modalityCounts.video]
        .every(value => Number.isSafeInteger(value) && value >= 0)) {
    throw new Error("Runtime media bundle provenance is invalid.");
  }
  const seen = new Set<string>();
  const counts: Record<PluginRuntimeMediaKind, number> = { image: 0, audio: 0, video: 0 };
  for (let ordinal = 0; ordinal < provenance.assets.length; ordinal += 1) {
    const asset = provenance.assets[ordinal], supported = asset ? SUPPORTED_MEDIA.get(asset.mimeType) : undefined;
    if (!asset || Object.keys(asset).sort().join("\0") !== ["assetId", "bytes", "kind", "memberName", "mimeType", "ordinal", "sha256"].join("\0") ||
        asset.ordinal !== ordinal || !SAFE_ID.test(asset.assetId) || seen.has(asset.assetId) || !supported || supported.kind !== asset.kind ||
        asset.memberName !== `media/${ordinal.toString().padStart(8, "0")}.${supported.extension}` || !Number.isSafeInteger(asset.bytes) || asset.bytes < 1 ||
        !SHA256.test(asset.sha256)) throw new Error("Runtime media bundle has invalid ordered asset provenance.");
    seen.add(asset.assetId);
    counts[asset.kind] += 1;
  }
  if (canonical(counts) !== canonical(provenance.modalityCounts) || sha256(canonical(provenance.assets)) !== provenance.assetManifestSha256 ||
      counts.image + counts.audio + counts.video !== provenance.assets.length) throw new Error("Runtime media bundle provenance counters or manifest hash do not close.");
  const lease = await controls.admitMemory(ARCHIVE_WORKING_BYTES, controls.signal);
  try {
    if (await regularFileSize(bundlePath) !== provenance.bundleBytes || await fileSha256(bundlePath, controls.signal) !== provenance.bundleSha256) {
      throw new Error("Runtime media bundle file does not match its exact byte/hash provenance.");
    }
    const extracted = await extractStreamingZip(bundlePath, resolve(bundlePath, ".."), {
      signal: controls.signal,
      checkDisk: (directory, bytes) => controls.checkDisk(directory, bytes, controls.signal)
    });
    try {
      if (extracted.entries.size !== provenance.assets.length) throw new Error("Runtime media bundle has an unexpected member count.");
      for (const asset of provenance.assets) {
        const entry = extracted.entries.get(asset.memberName);
        if (!entry || entry.uncompressedBytes !== asset.bytes || await fileSha256(entry.path, controls.signal) !== asset.sha256) {
          throw new Error(`Runtime media bundle member ${asset.memberName} failed exact byte/hash closure.`);
        }
      }
    } finally {
      await rm(extracted.root, { recursive: true, force: true });
    }
  } finally {
    await lease.release();
  }
}

/**
 * Transaction-owned binary custody for reviewed runtime-stream parsers.
 *
 * This helper does not invoke neural learning. It produces one verified,
 * media-only ZIP which the existing native archive ingestion path can visit in
 * ordinal order later, under the normal dataset cursor and training authority.
 */
export class PluginRuntimeMediaSpool {
  static async create(options: PluginRuntimeMediaSpoolOptions): Promise<PluginRuntimeMediaSpool> {
    if (!validIdentity(options.identity)) throw new Error("Runtime media parser identity is invalid.");
    if (!Number.isSafeInteger(options.maximumChunkBytes) || options.maximumChunkBytes < 1) {
      throw new Error("Runtime media parser chunk admission is invalid.");
    }
    const allowed = new Set<string>();
    for (const mediaType of options.allowedMediaTypes) {
      const normalized = normalizeSourceMediaType(mediaType);
      if (!normalized || normalized !== mediaType || !SUPPORTED_MEDIA.has(normalized)) {
        throw new Error(`Runtime media parser MIME ${String(mediaType)} is not a supported native modality input.`);
      }
      allowed.add(normalized);
    }
    if (!allowed.size) throw new Error("Runtime media parser declares no supported modality output MIME types.");
    options.signal?.throwIfAborted();
    const parent = join(resolve(options.workspaceRoot), "plugin-runtime-media");
    const root = join(parent, options.identity.transactionKey);
    await mkdir(parent, { recursive: true });
    await options.checkDisk(parent, 4_096, options.signal);
    await mkdir(root);
    return new PluginRuntimeMediaSpool(options, allowed, root);
  }

  private active?: ActiveAsset;
  private readonly assets: OwnedAsset[] = [];
  private terminal = false;
  private cleanupPromise?: Promise<void>;

  private constructor(
    private readonly options: PluginRuntimeMediaSpoolOptions,
    private readonly allowedMediaTypes: ReadonlySet<string>,
    private readonly root: string
  ) {}

  async start(input: { assetId: string; kind: PluginRuntimeMediaKind; mimeType: string }): Promise<void> {
    try {
      this.assertMutable();
      if (this.active) throw new Error("Runtime media parser must finish the current asset before starting another.");
      if (!SAFE_ID.test(input.assetId) || this.assets.some(asset => asset.assetId === input.assetId)) {
        throw new Error("Runtime media parser asset identity is invalid or repeated.");
      }
      const mimeType = normalizeSourceMediaType(input.mimeType);
      const supported = mimeType ? SUPPORTED_MEDIA.get(mimeType) : undefined;
      if (!mimeType || mimeType !== input.mimeType || !supported || supported.kind !== input.kind || !this.allowedMediaTypes.has(mimeType)) {
        throw new Error("Runtime media parser asset kind or MIME is outside its exact reviewed declaration.");
      }
      const ordinal = this.assets.length;
      if (!Number.isSafeInteger(ordinal)) throw new Error("Runtime media parser asset ordinal exceeds safe arithmetic.");
      const memberName = `media/${ordinal.toString().padStart(8, "0")}.${supported.extension}`;
      const finalPath = join(this.root, basename(memberName));
      const temporaryPath = `${finalPath}.part`;
      const metadataLease = await this.options.admitMemory(ASSET_METADATA_BYTES, this.options.signal);
      try {
        await this.options.checkDisk(this.root, 4_096, this.options.signal);
        const handle = await open(temporaryPath, constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY, 0o600);
        this.active = { ordinal, assetId: input.assetId, kind: input.kind, mimeType, memberName, temporaryPath,
          finalPath, handle, hash: createHash("sha256"), bytes: 0, prefix: Buffer.alloc(0), metadataLease };
      } catch (error) {
        await metadataLease.release();
        throw error;
      }
    } catch (error) {
      return this.fail(error);
    }
  }

  async write(input: { assetId: string; offset: number; bytesBase64: string }): Promise<void> {
    try {
      this.assertMutable();
      const active = this.requireActive(input.assetId);
      if (!Number.isSafeInteger(input.offset) || input.offset !== active.bytes) {
        throw new Error("Runtime media parser asset page is out of order.");
      }
      const length = decodedLength(input.bytesBase64);
      if (length > this.options.maximumChunkBytes) throw new Error("Runtime media parser asset page exceeds its admitted transport envelope.");
      if (!Number.isSafeInteger(active.bytes + length)) throw new Error("Runtime media parser asset size exceeds safe arithmetic.");
      const lease = await this.options.admitMemory(length, this.options.signal);
      try {
        this.options.signal?.throwIfAborted();
        await this.options.checkDisk(this.root, length, this.options.signal);
        const bytes = Buffer.from(input.bytesBase64, "base64");
        if (bytes.length !== length || bytes.toString("base64") !== input.bytesBase64) {
          throw new Error("Runtime media parser asset page is not canonical base64.");
        }
        await writeAll(active.handle, bytes, active.bytes);
        active.hash.update(bytes);
        if (active.prefix.length < MAGIC_PREFIX_BYTES) {
          active.prefix = Buffer.concat([active.prefix, bytes.subarray(0, MAGIC_PREFIX_BYTES - active.prefix.length)]);
        }
        active.bytes += bytes.length;
        this.options.signal?.throwIfAborted();
      } finally {
        await lease.release();
      }
    } catch (error) {
      return this.fail(error);
    }
  }

  async end(input: { assetId: string; bytes: number; sha256: string }): Promise<PluginRuntimeMediaAssetDescriptor> {
    try {
      this.assertMutable();
      const active = this.requireActive(input.assetId);
      if (!Number.isSafeInteger(input.bytes) || input.bytes < 1 || input.bytes !== active.bytes || !SHA256.test(input.sha256)) {
        throw new Error("Runtime media parser asset closure does not match its observed byte stream.");
      }
      const supported = SUPPORTED_MEDIA.get(active.mimeType)!;
      if (!supported.validMagic(active.prefix)) {
        throw new Error(`Runtime media parser asset ${active.assetId} does not match declared ${active.mimeType} container bytes.`);
      }
      const observedSha256 = active.hash.digest("hex");
      if (observedSha256 !== input.sha256) throw new Error("Runtime media parser asset checksum does not match its observed byte stream.");
      await active.handle.sync();
      await active.handle.close();
      await rename(active.temporaryPath, active.finalPath);
      const verificationLease = await this.options.admitMemory(128 * 1024, this.options.signal);
      try {
        if (await regularFileSize(active.finalPath) !== active.bytes || await fileSha256(active.finalPath, this.options.signal) !== observedSha256) {
          throw new Error("Runtime media parser asset failed committed file verification.");
        }
      } finally {
        await verificationLease.release();
      }
      const descriptor: PluginRuntimeMediaAssetDescriptor = { ordinal: active.ordinal, assetId: active.assetId, kind: active.kind,
        mimeType: active.mimeType, memberName: active.memberName, bytes: active.bytes, sha256: observedSha256 };
      this.assets.push({ ...descriptor, path: active.finalPath, metadataLease: active.metadataLease });
      this.active = undefined;
      return descriptor;
    } catch (error) {
      return this.fail(error);
    }
  }

  async complete(): Promise<PluginRuntimeMediaSpoolResult | PluginRuntimeMediaEmptyResult> {
    try {
      this.assertMutable();
      if (this.active) throw new Error("Runtime media parser ended before its current asset was closed.");
      this.terminal = true;
      if (!this.assets.length) {
        return { state: "no-assets", assets: [], assetManifestSha256: sha256(canonical([])),
          modalityCounts: { image: 0, audio: 0, video: 0 }, reason: "no-assets-no-modality-updates", cleanup: () => this.cleanup() };
      }
      const workLease = await this.options.admitMemory(ARCHIVE_WORKING_BYTES, this.options.signal);
      const bundlePath = join(this.root, "media-assets.zip");
      try {
        for (const asset of this.assets) {
          this.options.signal?.throwIfAborted();
          if (await regularFileSize(asset.path) !== asset.bytes || await fileSha256(asset.path, this.options.signal) !== asset.sha256) {
            throw new Error("Runtime media parser asset changed before bundle custody was committed.");
          }
        }
        await writeStreamingZip(bundlePath, this.assets.map(asset => ({ name: asset.memberName, sourcePath: asset.path })), {
          signal: this.options.signal,
          checkDisk: (directory, bytes) => this.options.checkDisk(directory, bytes, this.options.signal)
        });
        const bundleBytes = (await stat(bundlePath)).size;
        const bundleSha256 = await fileSha256(bundlePath, this.options.signal);
        const assets = this.assets.map(({ path: _path, metadataLease: _lease, ...asset }) => asset);
        const modalityCounts = assets.reduce<Record<PluginRuntimeMediaKind, number>>((counts, asset) => {
          counts[asset.kind] += 1;
          return counts;
        }, { image: 0, audio: 0, video: 0 });
        const provenance: PluginRuntimeMediaBundleProvenance = { format: "omni-runtime-parser-media-bundle", version: 1,
          ...this.options.identity, assets, assetManifestSha256: sha256(canonical(assets)), modalityCounts,
          bundleBytes, bundleSha256, complete: true };
        await verifyPluginRuntimeMediaBundle(bundlePath, provenance, this.options);
        return { state: "ready", bundlePath, provenance, cleanup: () => this.cleanup() };
      } finally {
        await workLease.release();
      }
    } catch (error) {
      return this.fail(error);
    }
  }

  async abort(): Promise<void> {
    this.terminal = true;
    await this.cleanup();
  }

  private assertMutable(): void {
    this.options.signal?.throwIfAborted();
    if (this.terminal) throw new Error("Runtime media parser custody is already closed.");
  }

  private requireActive(assetId: string): ActiveAsset {
    if (!this.active || this.active.assetId !== assetId) throw new Error("Runtime media parser asset page does not match the open asset.");
    return this.active;
  }

  private async fail(error: unknown): Promise<never> {
    this.terminal = true;
    await this.cleanup();
    throw error;
  }

  private cleanup(): Promise<void> {
    this.cleanupPromise ??= (async () => {
      const active = this.active;
      this.active = undefined;
      if (active) {
        await active.handle.close().catch(() => undefined);
        await Promise.resolve(active.metadataLease.release()).catch(() => undefined);
      }
      for (const asset of this.assets.splice(0)) await Promise.resolve(asset.metadataLease.release()).catch(() => undefined);
      await rm(this.root, { recursive: true, force: true });
    })();
    return this.cleanupPromise;
  }
}
