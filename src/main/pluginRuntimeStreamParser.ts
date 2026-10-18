import { createHash, randomUUID } from "node:crypto";
import { constants, createWriteStream, type WriteStream } from "node:fs";
import { lstat, mkdir, open, rename, rm, type FileHandle } from "node:fs/promises";
import { basename, join } from "node:path";
import { once } from "node:events";
import { finished } from "node:stream/promises";
import { requireDiskWrite } from "./diskSpace";
import { writeMutableFileIsolated } from "./mutableFileIsolation";
import { normalizeSourceMediaType } from "../shared/sourceMediaType";
import {
  PLUGIN_RUNTIME_PARSER_ABI,
  type PluginParserExecutionProvenance,
  type PluginPreparedTrainingSource,
  type PluginRuntimeMediaOutputDeclaration,
  type PluginRuntimeStreamParserRunner
} from "../shared/plugins";
import type {
  PluginRuntimeEvent,
  PluginRuntimeParserProgress,
  PluginRuntimeParserStep,
  PluginRuntimeParserStepResult,
  PluginRuntimePendingApproval,
  PluginRuntimeSession,
  PluginRuntimeTrainingParserActivationRequest
} from "../shared/pluginRuntime";
import { taskAttachmentFileIdentity } from "./taskAttachmentReadVerifier";
import { PluginRuntimeMediaSpool, verifyPluginRuntimeMediaBundle,
  type PluginRuntimeMediaBundleProvenance, type PluginRuntimeMediaEmptyResult,
  type PluginRuntimeMediaSpoolResult } from "./pluginRuntimeMediaSpool";

const SHA256 = /^[a-f0-9]{64}$/;
const DEFAULT_CHUNK_BYTES = 256 * 1024;
const FORBIDDEN_KEYS = new Set(["__proto__", "prototype", "constructor"]);
const parserRuns = new Map<string, Promise<void>>();
const parserConsumers = new Map<string, number>();

export interface PluginRuntimeTrainingParserHost {
  activateTrainingParser(request: PluginRuntimeTrainingParserActivationRequest): Promise<{ session: PluginRuntimeSession; maxMessageBytes: number }>;
  parserStep(brainId: string, sessionId: string, step: PluginRuntimeParserStep, signal: AbortSignal): Promise<PluginRuntimeParserStepResult>;
  approvals(brainId: string): Promise<PluginRuntimePendingApproval[]>;
  onEvent(listener: (event: PluginRuntimeEvent) => void): () => void;
  stop(brainId: string, sessionId: string): Promise<PluginRuntimeSession>;
  admitParserRead(brainId: string, bytes: number): Promise<{ release(): void | Promise<void> }>;
}

interface RuntimeParserReceipt {
  format: "omni-runtime-parser-spool";
  version: 1;
  transactionKey: string;
  outputFile: string;
  provenance: PluginParserExecutionProvenance;
  /** Full bounded-by-admission asset closure; absent preserves the JSONL v1 bytes. */
  media?: PluginRuntimeMediaBundleProvenance;
  receiptSha256: string;
}

function canonical(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  return `{${Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(",")}}`;
}
function sha256(value: string | Buffer): string { return createHash("sha256").update(value).digest("hex"); }
function receiptHash(value: Omit<RuntimeParserReceipt, "receiptSha256">): string { return sha256(canonical(value)); }
function safeJsonValue(value: unknown, depth = 0): void {
  if (depth > 64) throw new Error("Runtime parser output record is nested too deeply.");
  if (value === null || typeof value === "string" || typeof value === "boolean") return;
  if (typeof value === "number") { if (!Number.isFinite(value)) throw new Error("Runtime parser output contains a non-finite number."); return; }
  if (Array.isArray(value)) { value.forEach(child => safeJsonValue(child, depth + 1)); return; }
  if (!value || typeof value !== "object") throw new Error("Runtime parser output is not JSON data.");
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    if (FORBIDDEN_KEYS.has(key) || key.includes("\0")) throw new Error("Runtime parser output contains an unsafe key.");
    safeJsonValue(child, depth + 1);
  }
}
async function withGuardedFile<T>(path: string, expectedBytes: number, signal: AbortSignal | undefined,
  operation: (file: FileHandle, assertCurrent: () => Promise<void>) => Promise<T>): Promise<T> {
  signal?.throwIfAborted();
  const atPath = await lstat(path, { bigint: true });
  if (!atPath.isFile() || atPath.isSymbolicLink() || atPath.size !== BigInt(expectedBytes)) throw new Error("Runtime parser source is not the exact committed regular file.");
  const identity = taskAttachmentFileIdentity(atPath), noFollow = process.platform === "win32" ? 0 : constants.O_NOFOLLOW ?? 0;
  const file = await open(path, constants.O_RDONLY | noFollow);
  const assertCurrent = async (): Promise<void> => {
    signal?.throwIfAborted(); const [opened, current] = await Promise.all([file.stat({ bigint: true }), lstat(path, { bigint: true })]);
    if (!current.isFile() || current.isSymbolicLink() || taskAttachmentFileIdentity(opened) !== identity || taskAttachmentFileIdentity(current) !== identity) {
      throw new Error("Runtime parser source path or opened file changed during traversal.");
    }
  };
  try { await assertCurrent(); const result = await operation(file, assertCurrent); await assertCurrent(); return result; }
  finally { await file.close().catch(() => undefined); }
}
async function guardedFileSha256(path: string, bytes: number, signal?: AbortSignal): Promise<string> {
  return withGuardedFile(path, bytes, signal, async (file, assertCurrent) => {
    const hash = createHash("sha256"), scratch = Buffer.alloc(Math.min(64 * 1024, Math.max(1, bytes))); let offset = 0;
    while (offset < bytes) { signal?.throwIfAborted(); const row = await file.read(scratch, 0, Math.min(scratch.length, bytes - offset), offset);
      if (!row.bytesRead) throw new Error("Runtime parser guarded file ended before complete hashing."); hash.update(scratch.subarray(0, row.bytesRead)); offset += row.bytesRead; await assertCurrent(); }
    return hash.digest("hex");
  });
}
async function admittedFileSha256(host: PluginRuntimeTrainingParserHost, brainId: string, path: string, bytes: number, signal?: AbortSignal): Promise<string> {
  const lease = await host.admitParserRead(brainId, 128 * 1024);
  try { return await guardedFileSha256(path, bytes, signal); }
  finally { await lease.release(); }
}
async function guardedFileBytes(path: string, bytes: number, signal?: AbortSignal): Promise<Buffer> {
  return withGuardedFile(path, bytes, signal, async (file, assertCurrent) => {
    const result = Buffer.alloc(bytes); let offset = 0;
    while (offset < bytes) { signal?.throwIfAborted(); const row = await file.read(result, offset, bytes - offset, offset);
      if (!row.bytesRead) throw new Error("Runtime parser guarded file ended before complete reading."); offset += row.bytesRead; await assertCurrent(); }
    return result;
  });
}
async function validateJsonlFile(host: PluginRuntimeTrainingParserHost, brainId: string, path: string, bytes: number, signal?: AbortSignal): Promise<number> {
  const scanLease = await host.admitParserRead(brainId, 128 * 1024);
  try {
    return await withGuardedFile(path, bytes, signal, async (file, assertCurrent) => {
      const scratch = Buffer.alloc(64 * 1024); let cursor = 0, records = 0;
      while (cursor < bytes) {
        let scan = cursor, end = -1;
        while (scan < bytes && end < 0) {
          signal?.throwIfAborted(); const row = await file.read(scratch, 0, Math.min(scratch.length, bytes - scan), scan);
          if (!row.bytesRead) throw new Error("Runtime parser JSONL ended before its declared output bytes.");
          const newline = scratch.subarray(0, row.bytesRead).indexOf(0x0a); if (newline >= 0) end = scan + newline; else scan += row.bytesRead;
          await assertCurrent();
        }
        if (end < 0) throw new Error("Runtime parser output ended inside a JSONL record.");
        const length = end - cursor; if (length < 1) throw new Error("Runtime parser emitted an empty JSONL record.");
        const footprint = length * 12 + 65_536; if (!Number.isSafeInteger(footprint)) throw new Error("Runtime parser record exceeds safe physical admission arithmetic.");
        const recordLease = await host.admitParserRead(brainId, footprint);
        try {
          const raw = Buffer.alloc(length); let offset = 0;
          while (offset < length) { signal?.throwIfAborted(); const row = await file.read(raw, offset, length - offset, cursor + offset);
            if (!row.bytesRead) throw new Error("Runtime parser JSONL record was truncated during validation."); offset += row.bytesRead; await assertCurrent(); }
          const text = new TextDecoder("utf-8", { fatal: true }).decode(raw); if (!text.trim()) throw new Error("Runtime parser emitted an empty JSONL record.");
          const value = JSON.parse(text) as unknown; if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Runtime parser JSONL records must be objects."); safeJsonValue(value); records++;
        } finally { await recordLease.release(); }
        cursor = end + 1;
      }
      return records;
    });
  } finally { await scanLease.release(); }
}
function decodedOutput(value: string, maximum: number): Buffer {
  if (!value || value.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(value)) throw new Error("Runtime parser returned malformed base64 output.");
  const bytes = Buffer.from(value, "base64"); if (bytes.toString("base64") !== value || bytes.length > maximum) throw new Error("Runtime parser returned more than its admitted output page."); return bytes;
}
function decodedOutputLength(value: string): number {
  const padding = value.endsWith("==") ? 2 : value.endsWith("=") ? 1 : 0;
  return value.length / 4 * 3 - padding;
}

async function existingPrepared(host: PluginRuntimeTrainingParserHost, input: Parameters<PluginRuntimeStreamParserRunner["prepare"]>[0], outputPath: string, receiptPath: string): Promise<PluginPreparedTrainingSource | undefined> {
    let receiptInfo;
    try { receiptInfo = await lstat(receiptPath); }
    catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined; throw error; }
    if (!receiptInfo.isFile() || receiptInfo.isSymbolicLink() || !Number.isSafeInteger(receiptInfo.size) || receiptInfo.size < 1) throw new Error("Runtime parser cache receipt is not a safe bounded regular file.");
    const lease = await host.admitParserRead(input.brainId, receiptInfo.size);
    let text: string;
    try { text = (await guardedFileBytes(receiptPath, receiptInfo.size, input.request.signal)).toString("utf8"); }
    finally { await lease.release(); }
    const raw = JSON.parse(text) as RuntimeParserReceipt, { receiptSha256, ...body } = raw;
    const provenanceKeys = ["abi", "complete", "format", "formatId", "moduleId", "moduleSha256", "outputBytes", "outputRecords", "outputSha256",
      "packageSha256", "pluginId", "sourceBytes", "sourceSha256", "version", ...(raw.provenance?.inputMediaType === undefined ? [] : ["inputMediaType"])];
    if (!raw || Object.keys(raw).sort().join("\0") !== ["format", "outputFile", "provenance", "receiptSha256", "transactionKey", "version"].sort().join("\0") ||
        !raw.provenance || Object.keys(raw.provenance).sort().join("\0") !== provenanceKeys.sort().join("\0") || canonical(raw) !== text.trim() ||
        raw.format !== "omni-runtime-parser-spool" || raw.version !== 1 || raw.transactionKey !== input.request.transactionKey || raw.outputFile !== basename(outputPath) ||
        !SHA256.test(receiptSha256) || receiptHash(body) !== receiptSha256 || raw.provenance.format !== "omni-plugin-parser-execution" || raw.provenance.version !== 1 ||
        raw.provenance.abi !== PLUGIN_RUNTIME_PARSER_ABI || raw.provenance.pluginId !== input.binding.pluginId ||
        raw.provenance.packageSha256 !== input.binding.contentSha256 || raw.provenance.formatId !== input.binding.formatId || raw.provenance.moduleId !== input.module.id ||
        raw.provenance.moduleSha256 !== input.moduleSha256 || raw.provenance.sourceSha256 !== input.request.sourceSha256 ||
        raw.provenance.sourceBytes !== input.request.sourceBytes || raw.provenance.inputMediaType !== input.request.inputMediaType || raw.provenance.complete !== true ||
        !SHA256.test(raw.provenance.packageSha256) || !SHA256.test(raw.provenance.moduleSha256) || !SHA256.test(raw.provenance.sourceSha256) || !SHA256.test(raw.provenance.outputSha256) ||
        !Number.isSafeInteger(raw.provenance.outputBytes) || raw.provenance.outputBytes < 0 || !Number.isSafeInteger(raw.provenance.outputRecords) || raw.provenance.outputRecords < 0) throw new Error("Runtime parser cache receipt failed exact canonical identity validation.");
    const output = await lstat(outputPath), source = await lstat(input.request.sourcePath);
    if (!output.isFile() || output.isSymbolicLink() || output.nlink !== 1 || output.size !== raw.provenance.outputBytes || !source.isFile() || source.isSymbolicLink() ||
        source.size !== input.request.sourceBytes || await admittedFileSha256(host, input.brainId, outputPath, raw.provenance.outputBytes, input.request.signal) !== raw.provenance.outputSha256 ||
        await admittedFileSha256(host, input.brainId, input.request.sourcePath, input.request.sourceBytes, input.request.signal) !== input.request.sourceSha256) throw new Error("Runtime parser cache bytes no longer match their exact source/output receipt.");
    return { path: outputPath, parserFormat: "jsonl", provenance: raw.provenance,
      cleanup: async () => { await Promise.all([rm(outputPath, { force: true }), rm(receiptPath, { force: true })]); } };
}

function mediaIdentityMatches(
  media: PluginRuntimeMediaBundleProvenance,
  input: Parameters<PluginRuntimeStreamParserRunner["prepare"]>[0]
): boolean {
  return media.pluginId === input.binding.pluginId && media.packageSha256 === input.binding.contentSha256 &&
    media.formatId === input.binding.formatId && media.moduleId === input.module.id && media.moduleSha256 === input.moduleSha256 &&
    media.transactionKey === input.request.transactionKey && media.sourceSha256 === input.request.sourceSha256 &&
    media.sourceBytes === input.request.sourceBytes;
}

async function existingMediaPrepared(
  host: PluginRuntimeTrainingParserHost,
  input: Parameters<PluginRuntimeStreamParserRunner["prepare"]>[0],
  bundlePath: string,
  receiptPath: string
): Promise<PluginPreparedTrainingSource | undefined> {
  let receiptInfo;
  try { receiptInfo = await lstat(receiptPath); }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined; throw error; }
  if (!receiptInfo.isFile() || receiptInfo.isSymbolicLink() || !Number.isSafeInteger(receiptInfo.size) || receiptInfo.size < 1) {
    throw new Error("Runtime media parser cache receipt is not a safe admitted regular file.");
  }
  const receiptFootprint = receiptInfo.size * 12 + 65_536;
  if (!Number.isSafeInteger(receiptFootprint)) throw new Error("Runtime media parser cache receipt exceeds safe admission arithmetic.");
  const receiptLease = await host.admitParserRead(input.brainId, receiptFootprint);
  try {
  const text = (await guardedFileBytes(receiptPath, receiptInfo.size, input.request.signal)).toString("utf8");
  const raw = JSON.parse(text) as RuntimeParserReceipt, { receiptSha256, ...body } = raw;
  const provenance = raw.provenance, media = raw.media;
  const mediaKeys = ["assetManifestSha256", "assets", "bundleBytes", "bundleSha256", "complete", "format", "formatId", "modalityCounts",
    "moduleId", "moduleSha256", "packageSha256", "pluginId", "sourceBytes", "sourceSha256", "transactionKey", "version"];
  const provenanceKeys = ["abi", "assetManifestSha256", "complete", "format", "formatId", "moduleId", "moduleSha256",
    "outputAssets", "outputBytes", "outputKind", "outputModalityCounts", "outputRecords", "outputSha256", "packageSha256", "pluginId",
    "sourceBytes", "sourceSha256", "version", ...(provenance?.inputMediaType === undefined ? [] : ["inputMediaType"])];
  if (!raw || Object.keys(raw).sort().join("\0") !== ["format", "media", "outputFile", "provenance", "receiptSha256", "transactionKey", "version"].sort().join("\0") ||
      !provenance || Object.keys(provenance).sort().join("\0") !== provenanceKeys.sort().join("\0") || !media ||
      Object.keys(media).sort().join("\0") !== mediaKeys.sort().join("\0") ||
      canonical(raw) !== text.trim() || raw.format !== "omni-runtime-parser-spool" || raw.version !== 1 ||
      raw.transactionKey !== input.request.transactionKey || !SHA256.test(receiptSha256) || receiptHash(body) !== receiptSha256 ||
      provenance.format !== "omni-plugin-parser-execution" || provenance.version !== 1 || provenance.abi !== PLUGIN_RUNTIME_PARSER_ABI ||
      provenance.pluginId !== input.binding.pluginId || provenance.packageSha256 !== input.binding.contentSha256 || provenance.formatId !== input.binding.formatId ||
      provenance.moduleId !== input.module.id || provenance.moduleSha256 !== input.moduleSha256 || provenance.sourceSha256 !== input.request.sourceSha256 ||
      provenance.sourceBytes !== input.request.sourceBytes || provenance.inputMediaType !== input.request.inputMediaType || provenance.outputKind !== "media-assets" ||
      provenance.outputAssets !== provenance.outputRecords || provenance.outputAssets !== media.assets.length ||
      provenance.assetManifestSha256 !== media.assetManifestSha256 || canonical(provenance.outputModalityCounts) !== canonical(media.modalityCounts) ||
      provenance.outputSha256 !== media.bundleSha256 || provenance.outputBytes !== media.bundleBytes || provenance.complete !== true || !mediaIdentityMatches(media, input)) {
    throw new Error("Runtime media parser cache receipt failed exact canonical identity validation.");
  }
  const source = await lstat(input.request.sourcePath);
  if (!source.isFile() || source.isSymbolicLink() || source.size !== input.request.sourceBytes ||
      await admittedFileSha256(host, input.brainId, input.request.sourcePath, input.request.sourceBytes, input.request.signal) !== input.request.sourceSha256) {
    throw new Error("Runtime media parser cache source no longer matches its exact receipt.");
  }
  const cleanup = async (): Promise<void> => { await Promise.all([
    rm(join(input.request.workspaceRoot, "plugin-runtime-media", input.request.transactionKey), { recursive: true, force: true }),
    rm(receiptPath, { force: true })
  ]); };
  if (media.assets.length === 0) {
    if (raw.outputFile !== "no-assets" || media.bundleBytes !== 0 || media.bundleSha256 !== sha256(Buffer.alloc(0)) ||
        media.assetManifestSha256 !== sha256(canonical([])) || canonical(media.modalityCounts) !== canonical({ image: 0, audio: 0, video: 0 })) {
      throw new Error("Runtime media parser empty receipt claims output bytes or modality updates.");
    }
    return { state: "no-assets", parserFormat: "archive",
      provenance: { ...provenance, outputKind: "media-assets", outputAssets: 0 }, cleanup };
  }
  if (raw.outputFile !== basename(bundlePath)) throw new Error("Runtime media parser cache names another output bundle.");
  await verifyPluginRuntimeMediaBundle(bundlePath, media, {
    signal: input.request.signal,
    admitMemory: bytes => host.admitParserRead(input.brainId, bytes),
    checkDisk: async (directory, bytes) => { await requireDiskWrite(directory, { operationWriteBytes: bytes }); }
  });
  return { path: bundlePath, parserFormat: "archive", provenance, cleanup };
  } finally {
    await receiptLease.release();
  }
}

export class PluginRuntimeStreamParser implements PluginRuntimeStreamParserRunner {
  constructor(private readonly host: PluginRuntimeTrainingParserHost,
    private readonly options: { createOutput?: typeof createWriteStream } = {}) {}

  async prepare(input: Parameters<PluginRuntimeStreamParserRunner["prepare"]>[0]): Promise<PluginPreparedTrainingSource> {
    const consumerKey = `${input.request.workspaceRoot}\0${input.request.transactionKey}`;
    parserConsumers.set(consumerKey, (parserConsumers.get(consumerKey) ?? 0) + 1);
    const prior = parserRuns.get(consumerKey) ?? Promise.resolve(); let releaseRun!: () => void;
    const gate = new Promise<void>(resolve => { releaseRun = resolve; }), tail = prior.then(() => gate); parserRuns.set(consumerKey, tail);
    let prepared: PluginPreparedTrainingSource;
    try { await prior; prepared = await this.prepareLocked(input); }
    catch (error) { const remaining = (parserConsumers.get(consumerKey) ?? 1) - 1; if (remaining > 0) parserConsumers.set(consumerKey, remaining); else parserConsumers.delete(consumerKey); throw error; }
    finally { releaseRun(); if (parserRuns.get(consumerKey) === tail) parserRuns.delete(consumerKey); }
    let released = false;
    return { ...prepared, cleanup: async () => { if (released) return; released = true; const remaining = (parserConsumers.get(consumerKey) ?? 1) - 1;
      if (remaining > 0) { parserConsumers.set(consumerKey, remaining); return; } parserConsumers.delete(consumerKey); await prepared.cleanup(); } };
  }

  private async prepareLocked(input: Parameters<PluginRuntimeStreamParserRunner["prepare"]>[0]): Promise<PluginPreparedTrainingSource> {
    const request = input.request;
    if (input.binding.executable?.kind !== "runtime-stream" || input.binding.executable.moduleId !== input.module.id ||
        input.binding.executable.moduleSha256 !== input.moduleSha256 || input.binding.executable.abi !== PLUGIN_RUNTIME_PARSER_ABI ||
        !SHA256.test(input.binding.contentSha256) || !SHA256.test(input.moduleSha256) || !SHA256.test(request.sourceSha256) || !SHA256.test(request.transactionKey) ||
        !Number.isSafeInteger(request.sourceBytes) || request.sourceBytes < 0 ||
        request.inputMediaType !== undefined && normalizeSourceMediaType(request.inputMediaType) !== request.inputMediaType) throw new Error("Runtime parser preparation identity is invalid.");
    if (input.binding.executable.outputFormat === "media-assets") {
      const mediaTypes = input.binding.executable.outputMediaTypes;
      if (input.binding.parserFormat !== "archive" || !Array.isArray(mediaTypes) || !mediaTypes.length) {
        throw new Error("Runtime media parser preparation lost its exact reviewed output declaration.");
      }
      return this.prepareMediaLocked(input, { kind: "media-assets", mediaTypes: [...mediaTypes] });
    }
    if (input.binding.executable.outputFormat !== "jsonl" || input.binding.parserFormat !== "jsonl" || input.binding.executable.outputMediaTypes !== undefined) {
      throw new Error("Runtime JSONL parser preparation lost its exact reviewed output declaration.");
    }
    const directory = join(request.workspaceRoot, "plugin-runtime-parser"), outputPath = join(directory, `${request.transactionKey}.jsonl`),
      receiptPath = join(directory, `${request.transactionKey}.receipt.json`);
    await mkdir(directory, { recursive: true, mode: 0o700 });
    const existing = await existingPrepared(this.host, input, outputPath, receiptPath); if (existing) return existing;
    await Promise.all([rm(outputPath, { force: true }), rm(receiptPath, { force: true })]);
    const temporary = join(directory, `.${request.transactionKey}.${randomUUID()}.incomplete`), outputDigest = createHash("sha256"), sourceDigest = createHash("sha256");
    let inputBytes = 0, outputBytes = 0, outputRecords = 0, sequence = 0, session: PluginRuntimeSession | undefined, output: WriteStream | undefined;
    let outputCompletion: Promise<void> | undefined, outputFailure: Error | undefined;
    let unsubscribe = (): void => undefined;
    let progressTail = Promise.resolve();
    const deliverProgress = (state: PluginRuntimeParserProgress["state"], message: string, approval?: PluginRuntimeParserProgress["approval"]): Promise<void> => {
      const detail: PluginRuntimeParserProgress = { format: "omni-runtime-parser-progress", version: 1, brainId: input.brainId,
        pluginId: input.binding.pluginId, contentSha256: input.binding.contentSha256, moduleId: input.module.id, formatId: input.binding.formatId,
        transactionKey: request.transactionKey, state, inputBytes, sourceBytes: request.sourceBytes, outputBytes, outputRecords, message, ...(approval ? { approval } : {}) };
      const fraction = state === "complete" ? 0.25 : request.sourceBytes ? Math.min(0.249, inputBytes / request.sourceBytes * 0.249) : 0;
      return Promise.resolve(request.onProgress?.(fraction, message, detail));
    };
    const progress = (state: PluginRuntimeParserProgress["state"], message: string, approval?: PluginRuntimeParserProgress["approval"]): Promise<void> => {
      const next = progressTail.then(() => deliverProgress(state, message, approval)); progressTail = next.catch(() => undefined); return next;
    };
    try {
      await progress("starting", "Starting the exact reviewed runtime parser module.");
      const activated = await this.host.activateTrainingParser({ brainId: input.brainId, pluginId: input.binding.pluginId,
        contentSha256: input.binding.contentSha256, moduleId: input.module.id, moduleSha256: input.moduleSha256, formatId: input.binding.formatId,
        transactionKey: request.transactionKey, sourceSha256: request.sourceSha256, sourceBytes: request.sourceBytes,
        ...(request.inputMediaType ? { inputMediaType: request.inputMediaType } : {}), expectedCatalogRevision: input.binding.catalogRevision,
        expectedInstanceRevision: input.binding.instanceRevision, requestId: request.transactionKey });
      session = activated.session;
      const framePayloadBytes = Math.floor(Math.max(0, activated.maxMessageBytes - 2048) * 3 / 8);
      const pageBytes = Math.min(DEFAULT_CHUNK_BYTES, framePayloadBytes), outputPageBytes = framePayloadBytes;
      if (pageBytes < 1 || outputPageBytes < 1) throw new Error("Runtime parser has no admitted source/output page headroom.");
      output = (this.options.createOutput ?? createWriteStream)(temporary, { flags: "wx", mode: 0o600 });
      outputCompletion = finished(output).catch(error => { outputFailure = error instanceof Error ? error : new Error(String(error)); });
      unsubscribe = this.host.onEvent(event => {
        if (event.brainId !== input.brainId || event.kind !== "approval" || event.session?.sessionId !== session?.sessionId) return;
        const next = progressTail.then(async () => {
          const pending = (await this.host.approvals(input.brainId)).find(value => value.sessionId === session?.sessionId);
          if (pending) await deliverProgress("waiting-for-approval", `Waiting for approval: ${pending.invocation.toolId} · ${pending.invocation.action}.`, {
            sessionId: pending.sessionId, rpcId: pending.rpcId, reviewId: pending.reviewId, executionId: pending.executionId,
            toolId: pending.invocation.toolId, action: pending.invocation.action });
        }); progressTail = next.catch(() => undefined);
      });
      const writeOutput = async (base64: string): Promise<void> => { const bytes = decodedOutput(base64, outputPageBytes);
        if (outputFailure) throw outputFailure;
        await requireDiskWrite(request.workspaceRoot, { operationWriteBytes: bytes.length }); if (!output!.write(bytes)) await once(output!, "drain");
        if (outputFailure) throw outputFailure;
        outputDigest.update(bytes); outputBytes += bytes.length; };
      const step = async (operation: "push" | "drain", inputBase64: string | undefined, final: boolean | undefined, inputSequence: number): Promise<PluginRuntimeParserStepResult> => {
        if (outputFailure) throw outputFailure;
        request.signal?.throwIfAborted(); const id = sha256([request.transactionKey, operation, inputSequence, sequence++].join("\0"));
        const result = await this.host.parserStep(input.brainId, session!.sessionId, { id, operation, sequence: inputSequence, maxOutputBytes: outputPageBytes,
          ...(inputBase64 === undefined ? {} : { inputBase64 }), ...(final === undefined ? {} : { final }) }, request.signal ?? new AbortController().signal);
        if (outputFailure) throw outputFailure; return result;
      };
      const push = async (bytes: Buffer, final: boolean, inputSequence: number): Promise<void> => {
        let result = await step("push", bytes.toString("base64"), final, inputSequence);
        while (result.state === "output") { await writeOutput(result.outputBase64!); result = await step("drain", undefined, undefined, inputSequence); }
        if (final ? result.state !== "done" : result.state !== "need-input") throw new Error(final ? "Runtime parser did not finish after its final source page." : "Runtime parser requested an invalid source transition.");
      };
      await withGuardedFile(request.sourcePath, request.sourceBytes, request.signal, async (file, assertCurrent) => {
        let inputSequence = 0;
        const source = file.createReadStream({ autoClose: false, start: 0, highWaterMark: pageBytes, ...(request.signal ? { signal: request.signal } : {}) });
        for await (const value of source) { request.signal?.throwIfAborted(); await assertCurrent(); const bytes = Buffer.from(value as Buffer);
          sourceDigest.update(bytes); inputBytes += bytes.length; await push(bytes, false, inputSequence++);
          await progress("streaming", `Runtime parser streamed ${inputBytes.toLocaleString()} of ${request.sourceBytes.toLocaleString()} source bytes.`); }
        await assertCurrent();
        if (inputBytes !== request.sourceBytes || sourceDigest.digest("hex") !== request.sourceSha256) throw new Error("Runtime parser source changed or was not traversed completely.");
        await progress("finalizing", "Finalizing complete JSONL output and provenance."); await push(Buffer.alloc(0), true, inputSequence);
      });
      if (outputFailure) throw outputFailure; output.end(); await outputCompletion; if (outputFailure) throw outputFailure; output = undefined;
      outputRecords = await validateJsonlFile(this.host, input.brainId, temporary, outputBytes, request.signal);
      await rename(temporary, outputPath);
      const provenance: PluginParserExecutionProvenance = { format: "omni-plugin-parser-execution", version: 1, pluginId: input.binding.pluginId,
        packageSha256: input.binding.contentSha256, formatId: input.binding.formatId, moduleId: input.module.id, moduleSha256: input.moduleSha256,
        abi: PLUGIN_RUNTIME_PARSER_ABI, sourceSha256: request.sourceSha256, sourceBytes: request.sourceBytes,
        ...(request.inputMediaType ? { inputMediaType: request.inputMediaType } : {}), outputSha256: outputDigest.digest("hex"), outputBytes, outputRecords, complete: true };
      if (await admittedFileSha256(this.host, input.brainId, outputPath, outputBytes, request.signal) !== provenance.outputSha256) throw new Error("Runtime parser committed output failed its exact byte/hash verification.");
      const body: Omit<RuntimeParserReceipt, "receiptSha256"> = { format: "omni-runtime-parser-spool", version: 1,
        transactionKey: request.transactionKey, outputFile: basename(outputPath), provenance };
      const receiptBytes = Buffer.from(`${canonical({ ...body, receiptSha256: receiptHash(body) })}\n`, "utf8");
      await writeMutableFileIsolated(receiptPath, receiptBytes);
      if (!(await guardedFileBytes(receiptPath, receiptBytes.length, request.signal)).equals(receiptBytes)) throw new Error("Runtime parser receipt did not commit its exact bytes.");
      await this.host.stop(input.brainId, session.sessionId); session = undefined;
      await progress("complete", `Runtime parser visited all ${inputBytes.toLocaleString()} source bytes and committed ${outputRecords.toLocaleString()} records.`);
      return { path: outputPath, parserFormat: "jsonl", provenance,
        cleanup: async () => { await Promise.all([rm(outputPath, { force: true }), rm(receiptPath, { force: true })]); } };
    } catch (error) {
      const state = request.signal?.aborted ? "cancelled" : "failed"; await Promise.all([rm(outputPath, { force: true }), rm(receiptPath, { force: true })]).catch(() => undefined);
      await progress(state, error instanceof Error ? error.message : String(error)).catch(() => undefined); throw error;
    } finally {
      unsubscribe(); await progressTail.catch(() => undefined);
      if (output) { output.destroy(); await outputCompletion; }
      await rm(temporary, { force: true }).catch(() => undefined);
      if (session) await this.host.stop(input.brainId, session.sessionId).catch(() => undefined);
    }
  }

  private async prepareMediaLocked(
    input: Parameters<PluginRuntimeStreamParserRunner["prepare"]>[0],
    declaration: PluginRuntimeMediaOutputDeclaration
  ): Promise<PluginPreparedTrainingSource> {
    const request = input.request, receiptDirectory = join(request.workspaceRoot, "plugin-runtime-parser"),
      receiptPath = join(receiptDirectory, `${request.transactionKey}.receipt.json`),
      mediaDirectory = join(request.workspaceRoot, "plugin-runtime-media", request.transactionKey),
      bundlePath = join(mediaDirectory, "media-assets.zip");
    await mkdir(receiptDirectory, { recursive: true, mode: 0o700 });
    const existing = await existingMediaPrepared(this.host, input, bundlePath, receiptPath);
    if (existing) return existing;
    await Promise.all([rm(receiptPath, { force: true }), rm(mediaDirectory, { recursive: true, force: true })]);
    const sourceDigest = createHash("sha256");
    let inputBytes = 0, outputBytes = 0, outputRecords = 0, sequence = 0, session: PluginRuntimeSession | undefined;
    let spool: PluginRuntimeMediaSpool | undefined;
    let result: PluginRuntimeMediaSpoolResult | PluginRuntimeMediaEmptyResult | undefined;
    let unsubscribe = (): void => undefined;
    let progressTail = Promise.resolve();
    const deliverProgress = (state: PluginRuntimeParserProgress["state"], message: string, approval?: PluginRuntimeParserProgress["approval"]): Promise<void> => {
      const detail: PluginRuntimeParserProgress = { format: "omni-runtime-parser-progress", version: 1, brainId: input.brainId,
        pluginId: input.binding.pluginId, contentSha256: input.binding.contentSha256, moduleId: input.module.id, formatId: input.binding.formatId,
        transactionKey: request.transactionKey, state, inputBytes, sourceBytes: request.sourceBytes, outputBytes, outputRecords,
        outputAssets: outputRecords, message, ...(approval ? { approval } : {}) };
      const fraction = state === "complete" ? 0.25 : request.sourceBytes ? Math.min(0.249, inputBytes / request.sourceBytes * 0.249) : 0;
      return Promise.resolve(request.onProgress?.(fraction, message, detail));
    };
    const progress = (state: PluginRuntimeParserProgress["state"], message: string, approval?: PluginRuntimeParserProgress["approval"]): Promise<void> => {
      const next = progressTail.then(() => deliverProgress(state, message, approval)); progressTail = next.catch(() => undefined); return next;
    };
    try {
      await progress("starting", "Starting the exact reviewed runtime media parser module.");
      const activated = await this.host.activateTrainingParser({ brainId: input.brainId, pluginId: input.binding.pluginId,
        contentSha256: input.binding.contentSha256, moduleId: input.module.id, moduleSha256: input.moduleSha256, formatId: input.binding.formatId,
        transactionKey: request.transactionKey, sourceSha256: request.sourceSha256, sourceBytes: request.sourceBytes,
        ...(request.inputMediaType ? { inputMediaType: request.inputMediaType } : {}), output: declaration,
        expectedCatalogRevision: input.binding.catalogRevision, expectedInstanceRevision: input.binding.instanceRevision, requestId: request.transactionKey });
      session = activated.session;
      const framePayloadBytes = Math.floor(Math.max(0, activated.maxMessageBytes - 2048) * 3 / 8);
      const pageBytes = Math.min(DEFAULT_CHUNK_BYTES, framePayloadBytes), outputPageBytes = framePayloadBytes;
      if (pageBytes < 1 || outputPageBytes < 1) throw new Error("Runtime media parser has no admitted source/output page headroom.");
      spool = await PluginRuntimeMediaSpool.create({ workspaceRoot: request.workspaceRoot,
        identity: { pluginId: input.binding.pluginId, packageSha256: input.binding.contentSha256, formatId: input.binding.formatId,
          moduleId: input.module.id, moduleSha256: input.moduleSha256, transactionKey: request.transactionKey,
          sourceSha256: request.sourceSha256, sourceBytes: request.sourceBytes },
        allowedMediaTypes: declaration.mediaTypes, maximumChunkBytes: outputPageBytes, signal: request.signal,
        admitMemory: bytes => this.host.admitParserRead(input.brainId, bytes),
        checkDisk: async (directory, bytes) => { request.signal?.throwIfAborted(); await requireDiskWrite(directory, { operationWriteBytes: bytes }); request.signal?.throwIfAborted(); } });
      unsubscribe = this.host.onEvent(event => {
        if (event.brainId !== input.brainId || event.kind !== "approval" || event.session?.sessionId !== session?.sessionId) return;
        const next = progressTail.then(async () => {
          const pending = (await this.host.approvals(input.brainId)).find(value => value.sessionId === session?.sessionId);
          if (pending) await deliverProgress("waiting-for-approval", `Waiting for approval: ${pending.invocation.toolId} · ${pending.invocation.action}.`, {
            sessionId: pending.sessionId, rpcId: pending.rpcId, reviewId: pending.reviewId, executionId: pending.executionId,
            toolId: pending.invocation.toolId, action: pending.invocation.action });
        }); progressTail = next.catch(() => undefined);
      });
      const handleOutput = async (value: PluginRuntimeParserStepResult): Promise<boolean> => {
        if (value.state === "asset-start") { await spool!.start(value); return true; }
        if (value.state === "asset-chunk") { await spool!.write(value); const next = outputBytes + decodedOutputLength(value.bytesBase64);
          if (!Number.isSafeInteger(next)) throw new Error("Runtime media parser output bytes exceed safe progress arithmetic."); outputBytes = next; return true; }
        if (value.state === "asset-end") { await spool!.end(value); if (!Number.isSafeInteger(outputRecords + 1)) throw new Error("Runtime media parser asset count exceeds safe progress arithmetic."); outputRecords += 1; return true; }
        if (value.state === "output") throw new Error("Runtime media parser returned JSONL bytes instead of typed modality assets.");
        return false;
      };
      const step = async (operation: "push" | "drain", inputBase64: string | undefined, final: boolean | undefined, inputSequence: number): Promise<PluginRuntimeParserStepResult> => {
        request.signal?.throwIfAborted(); const id = sha256([request.transactionKey, operation, inputSequence, sequence++].join("\0"));
        return this.host.parserStep(input.brainId, session!.sessionId, { id, operation, sequence: inputSequence, maxOutputBytes: outputPageBytes,
          ...(inputBase64 === undefined ? {} : { inputBase64 }), ...(final === undefined ? {} : { final }) }, request.signal ?? new AbortController().signal);
      };
      const push = async (bytes: Buffer, final: boolean, inputSequence: number): Promise<void> => {
        let next = await step("push", bytes.toString("base64"), final, inputSequence);
        while (await handleOutput(next)) next = await step("drain", undefined, undefined, inputSequence);
        if (final ? next.state !== "done" : next.state !== "need-input") {
          throw new Error(final ? "Runtime media parser did not finish after its final source page." : "Runtime media parser requested an invalid source transition.");
        }
      };
      await withGuardedFile(request.sourcePath, request.sourceBytes, request.signal, async (file, assertCurrent) => {
        let inputSequence = 0;
        const source = file.createReadStream({ autoClose: false, start: 0, highWaterMark: pageBytes, ...(request.signal ? { signal: request.signal } : {}) });
        for await (const value of source) { request.signal?.throwIfAborted(); await assertCurrent(); const bytes = Buffer.from(value as Buffer);
          sourceDigest.update(bytes); inputBytes += bytes.length; await push(bytes, false, inputSequence++);
          await progress("streaming", `Runtime media parser streamed ${inputBytes.toLocaleString()} of ${request.sourceBytes.toLocaleString()} source bytes and closed ${outputRecords.toLocaleString()} assets.`); }
        await assertCurrent();
        if (inputBytes !== request.sourceBytes || sourceDigest.digest("hex") !== request.sourceSha256) throw new Error("Runtime media parser source changed or was not traversed completely.");
        await progress("finalizing", "Finalizing exact media assets, archive custody, and provenance."); await push(Buffer.alloc(0), true, inputSequence);
      });
      result = await spool.complete(); spool = undefined;
      const media: PluginRuntimeMediaBundleProvenance = result.state === "ready" ? result.provenance : {
        format: "omni-runtime-parser-media-bundle", version: 1, pluginId: input.binding.pluginId,
        packageSha256: input.binding.contentSha256, formatId: input.binding.formatId, moduleId: input.module.id,
        moduleSha256: input.moduleSha256, transactionKey: request.transactionKey, sourceSha256: request.sourceSha256,
        sourceBytes: request.sourceBytes, assets: [], assetManifestSha256: result.assetManifestSha256,
        modalityCounts: result.modalityCounts, bundleBytes: 0, bundleSha256: sha256(Buffer.alloc(0)), complete: true };
      outputBytes = media.bundleBytes; outputRecords = media.assets.length;
      const provenance: PluginParserExecutionProvenance = { format: "omni-plugin-parser-execution", version: 1,
        pluginId: input.binding.pluginId, packageSha256: input.binding.contentSha256, formatId: input.binding.formatId,
        moduleId: input.module.id, moduleSha256: input.moduleSha256, abi: PLUGIN_RUNTIME_PARSER_ABI,
        sourceSha256: request.sourceSha256, sourceBytes: request.sourceBytes,
        ...(request.inputMediaType ? { inputMediaType: request.inputMediaType } : {}), outputSha256: media.bundleSha256,
        outputBytes: media.bundleBytes, outputRecords: media.assets.length, outputKind: "media-assets", outputAssets: media.assets.length,
        assetManifestSha256: media.assetManifestSha256, outputModalityCounts: media.modalityCounts, complete: true };
      const body: Omit<RuntimeParserReceipt, "receiptSha256"> = { format: "omni-runtime-parser-spool", version: 1,
        transactionKey: request.transactionKey, outputFile: result.state === "ready" ? basename(result.bundlePath) : "no-assets", provenance, media };
      const receiptFootprint = media.assets.length * 1024 + 65_536;
      if (!Number.isSafeInteger(receiptFootprint)) throw new Error("Runtime media parser receipt exceeds safe admission arithmetic.");
      const receiptLease = await this.host.admitParserRead(input.brainId, receiptFootprint);
      try {
        const receiptBytes = Buffer.from(`${canonical({ ...body, receiptSha256: receiptHash(body) })}\n`, "utf8");
        await requireDiskWrite(receiptDirectory, { operationWriteBytes: receiptBytes.length });
        await writeMutableFileIsolated(receiptPath, receiptBytes);
        if (!(await guardedFileBytes(receiptPath, receiptBytes.length, request.signal)).equals(receiptBytes)) throw new Error("Runtime media parser receipt did not commit its exact bytes.");
      } finally { await receiptLease.release(); }
      await this.host.stop(input.brainId, session.sessionId); session = undefined;
      await progress("complete", result.state === "ready"
        ? `Runtime media parser visited all ${inputBytes.toLocaleString()} source bytes and committed ${outputRecords.toLocaleString()} native modality assets.`
        : `Runtime media parser visited all ${inputBytes.toLocaleString()} source bytes and emitted no assets; no modality updates will run.`);
      const cleanup = async (): Promise<void> => { await Promise.all([result!.cleanup(), rm(receiptPath, { force: true })]); };
      return result.state === "ready"
        ? { path: result.bundlePath, parserFormat: "archive", provenance, cleanup }
        : { state: "no-assets", parserFormat: "archive", provenance: { ...provenance, outputKind: "media-assets", outputAssets: 0 }, cleanup };
    } catch (error) {
      const state = request.signal?.aborted ? "cancelled" : "failed";
      await Promise.all([result?.cleanup(), spool?.abort(), rm(receiptPath, { force: true })]).catch(() => undefined);
      await progress(state, error instanceof Error ? error.message : String(error)).catch(() => undefined);
      throw error;
    } finally {
      unsubscribe(); await progressTail.catch(() => undefined);
      if (session) await this.host.stop(input.brainId, session.sessionId).catch(() => undefined);
    }
  }
}
