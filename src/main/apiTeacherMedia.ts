import { createHash, randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { lstat, mkdir, open, realpath, rename, rm } from "node:fs/promises";
import { dirname, isAbsolute, join, relative as pathRelative } from "node:path";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import { DatabaseSync } from "node:sqlite";
import type { ApiTeacherProvider } from "../shared/types";
import { apiTeacherLessonModalities, type ApiTeacherMediaLesson, type ApiTeacherLessonPart,
  type ApiTeacherMediaKind, type ApiTeacherMediaCoverage, type ApiTeacherLessonModality } from "../shared/apiTeacherMedia";
import type { ApiTeacherResponseEnvelope } from "../shared/apiTeacherMedia";
import type { ApiTeacherProviderPart } from "./apiTeacherMediaProviders";
import { sqliteFilename } from "./sqliteFilename";

export const API_TEACHER_MEDIA_REQUIRED_FEATURE = "api-teacher-typed-media-v1";
const SHA = /^[a-f0-9]{64}$/, ID = /^[A-Za-z0-9][A-Za-z0-9:._-]{0,255}$/;
const MEDIA = new Map<string, { kind: ApiTeacherMediaKind; extension: string; magic(value: Buffer): boolean }>([
  ["image/png", { kind: "image", extension: "png", magic: p => p.subarray(0, 8).equals(Buffer.from([137,80,78,71,13,10,26,10])) }],
  ["image/jpeg", { kind: "image", extension: "jpg", magic: p => p[0] === 255 && p[1] === 216 && p[2] === 255 }],
  ["image/webp", { kind: "image", extension: "webp", magic: p => p.toString("ascii",0,4) === "RIFF" && p.toString("ascii",8,12) === "WEBP" }],
  ["audio/wav", { kind: "audio", extension: "wav", magic: p => ["RIFF", "RF64"].includes(p.toString("ascii",0,4)) && p.toString("ascii",8,12) === "WAVE" }],
  ["audio/flac", { kind: "audio", extension: "flac", magic: p => p.toString("ascii",0,4) === "fLaC" }],
  ["audio/mpeg", { kind: "audio", extension: "mp3", magic: p => p.toString("ascii",0,3) === "ID3" || p[0] === 255 && (p[1]! & 224) === 224 }],
  ["video/mp4", { kind: "video", extension: "mp4", magic: p => p.toString("ascii",4,8) === "ftyp" }],
  ["video/webm", { kind: "video", extension: "webm", magic: p => p.subarray(0,4).equals(Buffer.from([26,69,223,163])) }]
]);
const record = (v: unknown): v is Record<string, unknown> => Boolean(v) && typeof v === "object" && !Array.isArray(v);
const hash = (v: string | Uint8Array): string => createHash("sha256").update(v).digest("hex");
function exact(value: unknown, keys: readonly string[]): Record<string, unknown> {
  if (!record(value) || Object.keys(value).some(key => !keys.includes(key)) || keys.some(key => !(key in value)))
    throw new Error("Typed teacher media has unknown or missing fields.");
  return value;
}
export interface ApiTeacherMediaIo {
  signal?: AbortSignal;
  admitMetadata?(bytes: number, purpose: string, signal?: AbortSignal): Promise<void> | void;
  checkDisk?(bytes: number, signal?: AbortSignal): Promise<void>;
}
export class ApiTeacherMediaResourcePause extends Error {
  readonly code = "API_TEACHER_MEDIA_RESOURCE_PAUSE";
}
function refuseTeacherEnvelopeAuthority(bytes: Buffer): string {
  const visible = bytes.toString("utf8").replace(/\\u([0-9a-f]{4})/gi, (_all, code: string) => String.fromCharCode(parseInt(code,16))).replace(/\\\//g,"/");
  if (/https?:\/\/|\bbearer(?:\s|\\[nt])+|\bsk-[A-Za-z0-9_-]{12,}|"(?:uri|url|authorization|x-api-key|x-goog-api-key|api_key|apiKey)"\s*:/i.test(visible))
    throw new Error("Inline teacher media refused a URL/auth-bearing envelope; it was not published or automatically retried.");
  return visible;
}
export async function admitApiTeacherMedia(bytes: number, purpose: string, io: ApiTeacherMediaIo): Promise<void> {
  io.signal?.throwIfAborted();
  if (!Number.isSafeInteger(bytes) || bytes < 0) throw new Error("Teacher media allocation is unsafe.");
  try { await io.admitMetadata?.(bytes, purpose, io.signal); }
  catch (error) { io.signal?.throwIfAborted(); throw new ApiTeacherMediaResourcePause(error instanceof Error ? error.message : String(error)); }
  io.signal?.throwIfAborted();
  if (bytes > freemem() || bytes > getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed)
    throw new ApiTeacherMediaResourcePause("Teacher media is waiting for physical/heap parsing RAM; received bytes were not retried or truncated.");
}
/** Actual bounded SQLite cache plus largest authenticated JSON-row allocation,
 * not a file-size cap or an invented share of the loaded brain's RAM. */
export async function admitApiTeacherDatabase(path: string, io: ApiTeacherMediaIo = {}): Promise<void> {
  await admitApiTeacherMedia(256 * 1024, "teacher custody SQLite cache", io);
  const info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("Teacher metadata database is not an owned regular file.");
  const database = new DatabaseSync(sqliteFilename(path), { readOnly: true, allowExtension: false });
  let largest = 0;
  try {
    database.exec("PRAGMA trusted_schema=OFF; PRAGMA cache_size=-256; PRAGMA mmap_size=0;");
    for (const table of ["jobs", "items", "history"] as const) {
      const bytes = Number(database.prepare(`SELECT COALESCE(MAX(length(CAST(payload AS BLOB))),0) AS bytes FROM ${table}`).get()?.bytes);
      if (!Number.isSafeInteger(bytes) || bytes < 0) throw new Error("Teacher metadata row length is invalid.");
      largest = Math.max(largest, bytes);
    }
  } finally { database.close(); }
  await admitApiTeacherMedia(largest * 12 + 65536, "teacher authenticated JSON-row parsing", io);
}
export function teacherMediaManifest(prompt: string, lesson: Pick<ApiTeacherMediaLesson, "provider" | "sourceEndpoint" | "model" | "deliveryId" | "parts">): string {
  const textBytes = lesson.parts.reduce((total, part) => total + (part.type === "text" ? Buffer.byteLength(part.text) : 0), Buffer.byteLength(prompt));
  const allocation = (textBytes + 4096 + lesson.parts.length * 512) * (lesson.parts.filter(part => part.type !== "text").length + 3) * 8;
  if (!Number.isSafeInteger(allocation) || allocation > freemem() || allocation > getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed)
    throw new ApiTeacherMediaResourcePause("Teacher ordered lesson manifest awaits physical/heap RAM; no data was truncated.");
  const condition = JSON.stringify({ input: prompt, annotations: lesson.parts.filter(part => part.type === "text")
    .map(part => ({ ordinal: part.ordinal, text: part.text, sha256: part.sha256 })) });
  const header = { text: JSON.stringify({ format: "omni-api-teacher-media-trajectory", version: 1,
    deliveryId: lesson.deliveryId, teacher: { provider: lesson.provider, model: lesson.model, endpoint: lesson.sourceEndpoint }, input: prompt,
    outputParts: lesson.parts.map(part => ({ ordinal: part.ordinal, type: part.type, sha256: part.sha256 })),
    hiddenBehavioralPrompt: false, rewardModel: false, rlhf: false }) };
  return [JSON.stringify(header), ...lesson.parts.map(part => JSON.stringify(part.type === "text" ? { text: part.text } : {
    format: "omni-teacher-media-pair-1", mediaPath: part.relativePath.slice("api-teachers/".length),
    mediaSha256: part.sha256, kind: part.type, mimeType: part.mimeType, conditioningText: condition
  }))].join("\n") + "\n";
}
export function validateApiTeacherMediaLesson(value: unknown, prompt: string, expectedDeliveryId?: string): ApiTeacherMediaLesson {
  const row = exact(value, ["format", "version", "provider", "sourceEndpoint", "model", "deliveryId", "responseModalities", "parts", "manifestPath", "manifestSha256"]);
  const endpoints: Record<string, string> = { openai: "https://api.openai.com/v1/responses", gemini: "https://generativelanguage.googleapis.com/v1beta/interactions",
    openrouter: "https://openrouter.ai/api/v1/images" };
  if (row.format !== "omni-api-teacher-media-lesson" || row.version !== 1 ||
      !["openai", "anthropic", "gemini", "openrouter", "ollama"].includes(String(row.provider)) ||
      row.sourceEndpoint !== endpoints[String(row.provider)] || !endpoints[String(row.provider)] ||
      typeof row.model !== "string" || !row.model || typeof row.deliveryId !== "string" || !ID.test(row.deliveryId) ||
      expectedDeliveryId !== undefined && row.deliveryId !== expectedDeliveryId || !Array.isArray(row.parts) || !row.parts.length ||
      typeof row.manifestSha256 !== "string" || !SHA.test(row.manifestSha256) ||
      row.manifestPath !== `api-teachers/lesson-${row.manifestSha256}.jsonl`) throw new Error("Typed teacher media identity is invalid.");
  const modalities = apiTeacherLessonModalities(row.responseModalities), seen = new Set<string>();
  row.parts.forEach((part, ordinal) => {
    if (!record(part) || part.ordinal !== ordinal || typeof part.sha256 !== "string" || !SHA.test(part.sha256))
      throw new Error("Typed teacher lesson order or hash is invalid.");
    if (part.type === "text") {
      exact(part, ["ordinal", "type", "text", "sha256"]);
      if (typeof part.text !== "string" || !part.text.trim() || hash(part.text) !== part.sha256)
        throw new Error("Typed teacher text checksum failed.");
    } else {
      exact(part, ["ordinal", "type", "mimeType", "relativePath", "bytes", "sha256"]);
      const supported = typeof part.mimeType === "string" ? MEDIA.get(part.mimeType) : undefined;
      if (!supported || supported.kind !== part.type || !modalities.includes(part.type as ApiTeacherMediaKind) ||
          !Number.isSafeInteger(part.bytes) || Number(part.bytes) < 1 ||
          part.relativePath !== `api-teachers/assets/${part.sha256}.${supported.extension}`)
        throw new Error("Typed teacher media path/MIME/modality is invalid.");
      seen.add(String(part.type));
    }
  });
  if (!modalities.some(kind => kind !== "text") || modalities.some(kind => kind !== "text" && !seen.has(kind)))
    throw new Error("Typed teacher lesson lacks actual requested media assets.");
  const lesson = row as unknown as ApiTeacherMediaLesson;
  if (hash(teacherMediaManifest(prompt, lesson)) !== lesson.manifestSha256) throw new Error("Typed teacher manifest checksum failed.");
  return structuredClone(lesson);
}
export function apiTeacherMediaFiles(lesson: ApiTeacherMediaLesson, prompt: string): Map<string, { sha256: string; bytes: number }> {
  validateApiTeacherMediaLesson(lesson, prompt);
  const files = new Map([[lesson.manifestPath, { sha256: lesson.manifestSha256, bytes: Buffer.byteLength(teacherMediaManifest(prompt, lesson)) }]]);
  for (const part of lesson.parts) if (part.type !== "text") files.set(part.relativePath, { sha256: part.sha256, bytes: part.bytes });
  return files;
}
export function safeApiTeacherMediaPath(path: string): boolean {
  return /^api-teachers\/(?:assets\/[a-f0-9]{64}\.(?:png|jpg|webp|wav|flac|mp3|mp4|webm)|lesson-[a-f0-9]{64}\.jsonl|envelopes\/[a-f0-9]{64}\.json)$/.test(path);
}
export function checkedApiTeacherResponseEnvelope(value: unknown): ApiTeacherResponseEnvelope {
  const row = exact(value, ["format", "version", "relativePath", "sha256", "bytes"]);
  if (row.format !== "omni-api-teacher-response-envelope" || row.version !== 1 || typeof row.sha256 !== "string" || !SHA.test(row.sha256) ||
      row.relativePath !== `api-teachers/envelopes/${row.sha256}.json` || !Number.isSafeInteger(row.bytes) || Number(row.bytes) < 2)
    throw new Error("Teacher received-response custody identity is invalid.");
  return structuredClone(row) as unknown as ApiTeacherResponseEnvelope;
}
/** Complete response custody precedes parsing admission; known keys are refused, never painted into binary assets. */
export async function storeApiTeacherResponseEnvelope(root: string, response: Response, secrets: readonly string[], io: ApiTeacherMediaIo): Promise<ApiTeacherResponseEnvelope> {
  root = await realpath(root);
  const needles = secrets.filter(Boolean).map(value => Buffer.from(value)), retained = Math.max(512, ...needles.map(value => value.byteLength * 6 + 64));
  await admitApiTeacherMedia(retained * 3 + 131072, "teacher response secret-safe streaming window", io);
  const reader = response.body?.getReader(); if (!reader) throw new Error("Teacher returned no media response body.");
  const result = await publish(root, "api-teachers/envelopes/{sha}.json", (async function*() {
    let tail = Buffer.alloc(0);
    try {
      for (;;) {
        const chunk = await reader.read(); if (chunk.done) break; if (!chunk.value) continue;
        for (let offset=0; offset<chunk.value.byteLength; offset+=65536) {
          const joined = Buffer.concat([tail, chunk.value.subarray(offset, offset+65536)]);
          if (needles.some(secret => joined.includes(secret))) throw new Error("Teacher response echoed a device credential; no media bytes were committed.");
          // First inline-only adapter refuses URL-bearing response envelopes,
          // not merely URLs selected for download. Never export signed handles.
          const visible = refuseTeacherEnvelopeAuthority(joined);
          if (secrets.some(secret => secret && visible.includes(secret))) throw new Error("Teacher response encoded a device credential; no media bytes were committed.");
          const emit = Math.max(0, joined.length - retained + 1);
          if (emit) yield joined.subarray(0, emit);
          tail = Buffer.from(joined.subarray(emit));
        }
      }
      if (needles.some(secret => tail.includes(secret))) throw new Error("Teacher response echoed a device credential.");
      if (tail.length) yield tail;
    } finally { await reader.cancel().catch(()=>undefined); reader.releaseLock(); }
  })(), io);
  return checkedApiTeacherResponseEnvelope({ format: "omni-api-teacher-response-envelope", version: 1,
    relativePath: `api-teachers/envelopes/${result.sha256}.json`, ...result });
}
export async function readApiTeacherResponseEnvelope(root: string, envelope: ApiTeacherResponseEnvelope, io: ApiTeacherMediaIo): Promise<Record<string, unknown>> {
  checkedApiTeacherResponseEnvelope(envelope); await verifyApiTeacherMediaFile(root, envelope.relativePath, envelope, io);
  await admitApiTeacherMedia(envelope.bytes * 8 + 65536, "teacher media JSON/inline response parsing", io);
  const handle = await open(join(await realpath(root), ...envelope.relativePath.split("/")), constants.O_RDONLY | (process.platform === "win32" ? 0 : constants.O_NOFOLLOW));
  try {
    const before = await handle.stat(), bytes = await handle.readFile();
    if (bytes.length !== envelope.bytes || hash(bytes) !== envelope.sha256 || !sameFile(before, await handle.stat()))
      throw new Error("Teacher received response changed during parsing.");
    const value: unknown = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
    if (!record(value)) throw new Error("Teacher media response is not a JSON object."); return value;
  } finally { await handle.close(); }
}
function sameFile(a: Awaited<ReturnType<typeof lstat>>, b: Awaited<ReturnType<typeof lstat>>): boolean {
  return a.dev === b.dev && a.ino === b.ino && a.size === b.size && a.mtimeMs === b.mtimeMs;
}
/** Open/FD/hash checks protect exact source bytes; signatures are not decoder-quality proof. */
export async function verifyApiTeacherMediaFile(root: string, relative: string, expected: { sha256: string; bytes: number }, io: ApiTeacherMediaIo = {}): Promise<void> {
  if (!safeApiTeacherMediaPath(relative)) throw new Error("Unsafe teacher media custody path.");
  io.signal?.throwIfAborted(); const canonical = await realpath(root), path = join(canonical, ...relative.split("/"));
  let parent = canonical;
  for (const piece of relative.split("/").slice(0,-1)) {
    parent = join(parent, piece); const info = await lstat(parent);
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Teacher media custody contains a linked directory.");
  }
  const before = await lstat(path);
  if (!before.isFile() || before.isSymbolicLink() || before.size !== expected.bytes) throw new Error("Teacher media custody byte count changed.");
  const handle = await open(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
  try {
    if (!sameFile(before, await handle.stat())) throw new Error("Teacher media custody was replaced while opening.");
    await admitApiTeacherMedia(65536, "teacher media hash window", io);
    if (relative.startsWith("api-teachers/assets/")) {
      const supported = [...MEDIA.values()].find(value => relative.endsWith("." + value.extension)), prefix = Buffer.alloc(64);
      const { bytesRead } = await handle.read(prefix, 0, 64, 0);
      if (!supported?.magic(prefix.subarray(0, bytesRead))) throw new Error("Teacher media custody container signature differs.");
    }
    const digest = createHash("sha256");
    let tail = Buffer.alloc(0);
    for await (const chunk of handle.createReadStream({ autoClose: false, highWaterMark: 65536 })) {
      io.signal?.throwIfAborted(); digest.update(chunk);
      if (relative.startsWith("api-teachers/envelopes/")) { const joined = Buffer.concat([tail, chunk]); refuseTeacherEnvelopeAuthority(joined); tail = Buffer.from(joined.subarray(-512)); }
    }
    if (digest.digest("hex") !== expected.sha256 || !sameFile(before, await handle.stat()) || !sameFile(before, await lstat(path)))
      throw new Error("Teacher media custody checksum or file identity changed.");
  } finally { await handle.close(); }
}
async function publish(root: string, relative: string, chunks: AsyncIterable<Uint8Array>, io: ApiTeacherMediaIo): Promise<{ sha256: string; bytes: number }> {
  const canonical = await realpath(root), directory = dirname(join(canonical, relative));
  let current = canonical;
  for (const piece of relative.split("/").slice(0,-1)) {
    if (!piece || piece === "." || piece === "..") throw new Error("Teacher media publication has an unsafe directory.");
    current = join(current, piece);
    let info = await lstat(current).catch((error: NodeJS.ErrnoException) => error.code === "ENOENT" ? undefined : Promise.reject(error));
    if (!info) { await mkdir(current, { mode: 0o700 }); info = await lstat(current); }
    if (!info.isDirectory() || info.isSymbolicLink() || await realpath(current) !== current)
      throw new Error("Teacher media publication directory is not owned.");
  }
  const canonicalDirectory = await realpath(directory);
  const owned = pathRelative(canonical, canonicalDirectory);
  if (canonicalDirectory !== directory || !owned || owned.startsWith("..") || isAbsolute(owned)) throw new Error("Teacher media publication directory is not owned.");
  const temporary = join(directory, `.teacher-${randomUUID()}.next`), output = await open(temporary, "wx", 0o600);
  let bytes = 0; const digest = createHash("sha256");
  try {
    for await (const chunk of chunks) {
      io.signal?.throwIfAborted(); await admitApiTeacherMedia(chunk.byteLength * 2 + 65536, "teacher media publication page", io);
      try { await io.checkDisk?.(chunk.byteLength + 65536, io.signal); }
      catch (error) { io.signal?.throwIfAborted(); throw new ApiTeacherMediaResourcePause(error instanceof Error ? error.message : String(error)); }
      io.signal?.throwIfAborted();
      let offset = 0; while (offset < chunk.byteLength) {
        const result = await output.write(chunk, offset, chunk.byteLength - offset, bytes + offset);
        if (!result.bytesWritten) throw new Error("Teacher media asset write did not progress."); offset += result.bytesWritten;
      }
      bytes += chunk.byteLength; digest.update(chunk);
    }
    await output.sync(); await output.close(); const sha256 = digest.digest("hex"), destination = join(root, relative.replace("{sha}", sha256));
    if (await lstat(destination).then(()=>true, error => error.code === "ENOENT" ? false : Promise.reject(error))) {
      await verifyApiTeacherMediaFile(root, relative.replace("{sha}", sha256), { sha256, bytes }, io);
    } else await rename(temporary, destination);
    return { sha256, bytes };
  } catch (error) {
    if (["ENOSPC", "EDQUOT", "ENOMEM"].includes(String((error as NodeJS.ErrnoException)?.code)))
      throw new ApiTeacherMediaResourcePause("Teacher media publication awaits physical RAM/disk resources; committed response custody was retained.");
    throw error;
  } finally { await output.close().catch(()=>undefined); await rm(temporary, { force: true }); }
}
export async function storeApiTeacherMediaLesson(root: string, input: { provider: ApiTeacherProvider; sourceEndpoint: string; model: string; deliveryId: string;
  prompt: string; responseModalities: ApiTeacherLessonModality[]; parts: ApiTeacherProviderPart[] }, io: ApiTeacherMediaIo = {}): Promise<ApiTeacherMediaLesson> {
  root = await realpath(root);
  const parts: ApiTeacherLessonPart[] = [];
  for (const part of input.parts) {
    io.signal?.throwIfAborted();
    if (part.type === "text") { parts.push({ ordinal: parts.length, type: "text", text: part.text, sha256: hash(part.text) }); continue; }
    const supported = MEDIA.get(part.mimeType);
    if (!supported || supported.kind !== part.type || !part.data || part.data.length % 4 || !/^[A-Za-z0-9+/]*={0,2}$/.test(part.data) ||
        Buffer.from(part.data.slice(-4), "base64").toString("base64") !== part.data.slice(-4))
      throw new Error("Teacher media needs canonical base64 and a supported typed container.");
    await admitApiTeacherMedia(131072, "teacher media decode window", io);
    if (!supported.magic(Buffer.from(part.data.slice(0,88), "base64"))) throw new Error("Teacher media MIME does not match its container signature.");
    const result = await publish(root, `api-teachers/assets/{sha}.${supported.extension}`, (async function*() {
      for (let offset=0; offset<part.data.length; offset+=65536) yield Buffer.from(part.data.slice(offset,offset+65536), "base64");
    })(), io);
    parts.push({ ordinal: parts.length, type: part.type, mimeType: part.mimeType, relativePath: `api-teachers/assets/${result.sha256}.${supported.extension}`, ...result });
  }
  const partial = { provider: input.provider, sourceEndpoint: input.sourceEndpoint, model: input.model, deliveryId: input.deliveryId, parts };
  const textBytes = parts.reduce((sum, part) => sum + (part.type === "text" ? Buffer.byteLength(part.text) : 0), Buffer.byteLength(input.prompt));
  await admitApiTeacherMedia((textBytes + 4096 + parts.length * 512) * (parts.filter(part=>part.type!=="text").length + 3) * 8, "teacher ordered manifest allocation", io);
  const manifest = teacherMediaManifest(input.prompt, partial);
  await admitApiTeacherMedia(Buffer.byteLength(manifest) * 4, "teacher media manifest", io);
  const result = await publish(root, "api-teachers/lesson-{sha}.jsonl", (async function*(){ yield Buffer.from(manifest); })(), io);
  return validateApiTeacherMediaLesson({ format: "omni-api-teacher-media-lesson", version: 1, ...partial,
    responseModalities: input.responseModalities, manifestPath: `api-teachers/lesson-${result.sha256}.jsonl`, manifestSha256: result.sha256 }, input.prompt, input.deliveryId);
}
export function checkedApiTeacherMediaCoverage(value: unknown, lesson: ApiTeacherMediaLesson): ApiTeacherMediaCoverage {
  const coverage = exact(value, ["complete", "manifestSha256", "byModality"]);
  if (coverage.complete !== true || coverage.manifestSha256 !== lesson.manifestSha256 || !record(coverage.byModality))
    throw new Error("Teacher media has no complete native learning coverage.");
  const expected = new Map<ApiTeacherMediaKind, number>();
  for (const part of lesson.parts) if (part.type !== "text") expected.set(part.type, (expected.get(part.type) ?? 0) + 1);
  for (const [kind, minimum] of expected) {
    const row = exact(coverage.byModality[kind], ["records", "windows", "processedUnits", "complete"]);
    if (row.complete !== true || ![row.records,row.windows,row.processedUnits].every(v=>Number.isSafeInteger(v) && Number(v)>0) || Number(row.records) < minimum)
      throw new Error(`Teacher ${kind} learning did not cover all real assets.`);
  }
  if (Object.keys(coverage.byModality).length !== expected.size || Object.keys(coverage.byModality).some(kind => !expected.has(kind as ApiTeacherMediaKind)))
    throw new Error("Teacher media coverage contains an unrequested modality.");
  return structuredClone(coverage) as unknown as ApiTeacherMediaCoverage;
}
