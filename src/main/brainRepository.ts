import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import {
  access,
  cp,
  copyFile,
  link,
  lstat,
  mkdir,
  mkdtemp,
  open,
  readFile,
  readdir,
  realpath,
  rename,
  rm,
  stat,
  writeFile
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { strToU8 } from "fflate";
import {
  BRAIN_SCHEMA_VERSION,
  DEFAULT_CONFIG,
  type BrainConfig,
  type BrainDocument,
  type BrainExportMode,
  type BrainMetrics,
  type BrainSnapshotSummary,
  type BrainSummary,
  type ToolPermissionRecord
} from "../shared/types";
import {
  assertSafeArchivePath,
  extractStreamingZip,
  streamFileSha256,
  writeStreamingZip,
  type ExtractedZipArchive,
  type StreamingZipSource
} from "./streamingZip";

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const BUNDLE_FORMAT = "omni-brain";
const BUNDLE_VERSION = 1;
/** Must remain identical to engine/omni_core/vsa.py's store format. */
export const SUBSTRATE_STORE_FORMAT = "omni-substrate-shards";
const STABLE_RELEASE_FORMAT = "stable-1.0";
const BETA_REVIEW_FILE = ".stable-v1-beta-review.json";

interface OmniManifest {
  format: typeof BUNDLE_FORMAT;
  formatVersion: number;
  releaseFormat: typeof STABLE_RELEASE_FORMAT;
  architecture: "OmniCortex";
  architectureSchemaVersion: number;
  exportedAt: string;
  brain: {
    id: string;
    name: string;
    lineage: BrainDocument["lineage"];
  };
  mode: "current-portable" | "origin-portable" | "private-archive" | "referenced-local";
  engineMaterialized: boolean;
  memoryRecipe: string;
  rawEpisodesPresent: boolean;
  quantization: "ternary-effective";
  packedTernary?: {
    format: "omni-packed-ternary";
    formatVersion: 1;
    currentManifestSha256: string;
    originManifestSha256: string;
    currentTensorCount: number;
    originTensorCount: number;
    references?: {
      current: Record<string, string>;
      origin: Record<string, string>;
    };
  };
  secretRedaction: {
    version: 1;
    replacements: number;
  };
  licenseLedger: {
    application: "PolyForm-Noncommercial-1.0.0-or-commercial-license";
    sources: Array<{
      name: string;
      provenanceUrl?: string;
      license: string;
      licenseUrl?: string;
    }>;
  };
  references?: {
    currentCore: string;
    currentPlasticity: string;
    originCore: string;
    originPlasticity: string;
  };
  files: Record<string, { sha256: string; bytes: number }>;
}

interface StreamingPackedTernaryDirectory {
  manifestSha256: string;
  tensorCount: number;
  files: Map<string, string>;
}

export interface ManagedBetaBrain {
  id: string;
  name: string;
  path: string;
  reason: "beta-document" | "beta-engine" | "invalid-document";
}

export const DEFAULT_TOOL_PERMISSIONS: ToolPermissionRecord[] = [
  "windows.files",
  "windows.powershell",
  "code.execute",
  "web.search",
  "web.fetch",
  "browser.automation",
  "modality.imagine",
  "agent.fork",
  "source.self-modify"
].map((toolId) => ({
  toolId,
  label: toolId
    .split(".")
    .map((part) => `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`)
    .join(" "),
  level:
    toolId === "browser.automation" || toolId === "source.self-modify"
      ? ("off" as const)
      : toolId === "modality.imagine"
        ? ("auto" as const)
        : ("ask" as const),
  updatedAt: new Date(0).toISOString()
}));

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

interface RedactionCounter {
  replacements: number;
}

const SECRET_FIELD =
  /^(?:password|passwd|passphrase|secret|client[_-]?secret|api[_-]?key|access[_-]?token|refresh[_-]?token|auth(?:orization)?|credential|cookie|session[_-]?cookie|private[_-]?key)$/i;

function redactSecretText(value: string, counter: RedactionCounter): string {
  let redacted = value;
  const patterns = [
    /-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----/g,
    /\bAKIA[0-9A-Z]{16}\b/g,
    /\bgh[pousr]_[A-Za-z0-9]{20,}\b/g,
    /\bsk-[A-Za-z0-9_-]{20,}\b/g,
    /\bBearer\s+[A-Za-z0-9._~+/-]{16,}=*\b/gi,
    /\b(api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|secret)\s*[:=]\s*["']?[^\s"',;]{8,}["']?/gi,
    /https?:\/\/[^:\s/@]{1,256}:[^@\s/]{1,256}@/gi
  ];
  for (const pattern of patterns) {
    redacted = redacted.replace(pattern, () => {
      counter.replacements += 1;
      return "[REDACTED_SECRET]";
    });
  }
  return redacted;
}

async function assertFileContainsNoPortableSecrets(
  path: string,
  label: string
): Promise<void> {
  const probe = await open(path, "r");
  try {
    const sample = Buffer.alloc(256 * 1024);
    const { bytesRead } = await probe.read(sample, 0, sample.byteLength, 0);
    const text = sample.subarray(0, bytesRead).toString("utf8");
    const binaryRatio =
      text.length === 0
        ? 0
        : ((text.match(/\uFFFD/g)?.length ?? 0) + (text.match(/\0/g)?.length ?? 0)) /
          text.length;
    if (binaryRatio >= 0.01) return;
  } finally {
    await probe.close();
  }
  let overlap = "";
  for await (const chunk of createReadStream(path, { encoding: "utf8" })) {
    const combined = overlap + chunk;
    const counter: RedactionCounter = { replacements: 0 };
    redactSecretText(combined, counter);
    if (counter.replacements > 0) {
      throw new Error(
        `${label} appears to contain credentials. Remove or sanitize it before a private archive export.`
      );
    }
    overlap = combined.slice(-4096);
  }
}

function redactPortableValue(
  value: unknown,
  counter: RedactionCounter,
  key = ""
): unknown {
  if (SECRET_FIELD.test(key) && value !== undefined && value !== null) {
    counter.replacements += 1;
    return "[REDACTED_SECRET]";
  }
  if (typeof value === "string") return redactSecretText(value, counter);
  if (Array.isArray(value)) {
    return value.map((entry) => redactPortableValue(entry, counter));
  }
  if (isRecord(value)) {
    return Object.fromEntries(
      Object.entries(value).map(([entryKey, entryValue]) => [
        entryKey,
        redactPortableValue(entryValue, counter, entryKey)
      ])
    );
  }
  return value;
}

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

async function fileSha256(path: string): Promise<string> {
  const digest = createHash("sha256");
  await new Promise<void>((resolveHash, rejectHash) => {
    const stream = createReadStream(path);
    stream.on("data", (chunk) => digest.update(chunk));
    stream.once("error", rejectHash);
    stream.once("end", resolveHash);
  });
  return digest.digest("hex");
}

function validEmptySafetensors(note: string): Uint8Array {
  const header = Buffer.from(
    JSON.stringify({ __metadata__: { format: "omni-empty", note } }).padEnd(256, " "),
    "utf8"
  );
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(header.byteLength));
  return new Uint8Array(Buffer.concat([prefix, header]));
}

function safeZipPath(path: string): boolean {
  if (
    !path ||
    path.includes("\0") ||
    path.includes("\\") ||
    path.startsWith("/") ||
    /^[a-zA-Z]:/.test(path)
  ) {
    return false;
  }
  const parts = path.split("/");
  return parts.every((part) => part !== "" && part !== "." && part !== "..");
}

function assertAllowedBundlePath(path: string): void {
  if (!safeZipPath(path)) throw new Error(`Unsafe path in .omni bundle: ${path}`);
  if (
    /\.(?:exe|dll|com|bat|cmd|ps1|msi|scr|js|jse|vbs|vbe|wsf|wsh|lnk|app|dylib|so|pyc)$/i.test(
      path
    )
  ) {
    throw new Error(`Executable content is not allowed in .omni bundles: ${path}`);
  }
}

function parseChecksumFile(value: string): Map<string, string> {
  const checksums = new Map<string, string>();
  for (const line of value.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const match = /^([a-f0-9]{64})  ([^\r\n]+)$/i.exec(line);
    if (!match?.[1] || !match[2]) throw new Error("checksums.sha256 has an invalid record.");
    assertAllowedBundlePath(match[2]);
    if (checksums.has(match[2])) throw new Error(`Duplicate checksum for ${match[2]}.`);
    checksums.set(match[2], match[1].toLocaleLowerCase());
  }
  return checksums;
}

function assertSafeTensors(contents: Uint8Array, label: string): void {
  const buffer = Buffer.from(contents.buffer, contents.byteOffset, contents.byteLength);
  if (buffer.byteLength < 10) throw new Error(`${label} is not a valid safetensors file.`);
  const headerLength = Number(buffer.readBigUInt64LE(0));
  if (!Number.isSafeInteger(headerLength) || headerLength < 2 || headerLength > buffer.byteLength - 8) {
    throw new Error(`${label} has an invalid safetensors header length.`);
  }
  let header: unknown;
  try {
    header = JSON.parse(buffer.subarray(8, 8 + headerLength).toString("utf8").trim());
  } catch {
    throw new Error(`${label} has an invalid safetensors JSON header.`);
  }
  if (!isRecord(header)) throw new Error(`${label} has an invalid safetensors header.`);
  const dataBytes = buffer.byteLength - 8 - headerLength;
  for (const [name, descriptor] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (!isRecord(descriptor) || !Array.isArray(descriptor.data_offsets)) {
      throw new Error(`${label} contains an invalid tensor descriptor.`);
    }
    const offsets = descriptor.data_offsets;
    if (
      offsets.length !== 2 ||
      !offsets.every((offset) => typeof offset === "number" && Number.isSafeInteger(offset)) ||
      (offsets[0] as number) < 0 ||
      (offsets[1] as number) < (offsets[0] as number) ||
      (offsets[1] as number) > dataBytes
    ) {
      throw new Error(`${label} contains out-of-bounds tensor data.`);
    }
  }
}

async function assertSafeTensorsFile(path: string, label: string): Promise<void> {
  const info = await stat(path);
  if (!info.isFile() || info.size < 10) {
    throw new Error(`${label} is not a valid safetensors file.`);
  }
  const handle = await open(path, "r");
  try {
    const prefix = Buffer.alloc(8);
    const prefixRead = await handle.read(prefix, 0, prefix.byteLength, 0);
    if (prefixRead.bytesRead !== prefix.byteLength) {
      throw new Error(`${label} is not a valid safetensors file.`);
    }
    const headerLength = Number(prefix.readBigUInt64LE(0));
    if (
      !Number.isSafeInteger(headerLength) ||
      headerLength < 2 ||
      headerLength > info.size - 8
    ) {
      throw new Error(`${label} has an invalid safetensors header length.`);
    }
    const headerBytes = Buffer.allocUnsafe(headerLength);
    let cursor = 0;
    while (cursor < headerLength) {
      const result = await handle.read(
        headerBytes,
        cursor,
        headerLength - cursor,
        8 + cursor
      );
      if (result.bytesRead <= 0) {
        throw new Error(`${label} has a truncated safetensors header.`);
      }
      cursor += result.bytesRead;
    }
    let header: unknown;
    try {
      header = JSON.parse(headerBytes.toString("utf8").trim());
    } catch {
      throw new Error(`${label} has an invalid safetensors JSON header.`);
    }
    if (!isRecord(header)) throw new Error(`${label} has an invalid safetensors header.`);
    const dataBytes = info.size - 8 - headerLength;
    for (const [name, descriptor] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      if (!isRecord(descriptor) || !Array.isArray(descriptor.data_offsets)) {
        throw new Error(`${label} contains an invalid tensor descriptor.`);
      }
      const offsets = descriptor.data_offsets;
      if (
        offsets.length !== 2 ||
        !offsets.every((offset) => typeof offset === "number" && Number.isSafeInteger(offset)) ||
        (offsets[0] as number) < 0 ||
        (offsets[1] as number) < (offsets[0] as number) ||
        (offsets[1] as number) > dataBytes
      ) {
        throw new Error(`${label} contains out-of-bounds tensor data.`);
      }
    }
  } finally {
    await handle.close();
  }
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.entries(value as Record<string, unknown>)
    // Python's json.dumps(sort_keys=True) compares Unicode code points.
    // localeCompare is locale-sensitive (and orders "+1" after "-1" on
    // some hosts), which would reject the worker's canonical manifests.
    .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
    .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`)
    .join(",")}}`;
}

function normalizeCanonicalJsonNumbers(value: string): string {
  let normalized = "";
  let inString = false;
  let escaped = false;
  for (let index = 0; index < value.length;) {
    const character = value[index]!;
    if (inString) {
      normalized += character;
      index += 1;
      if (escaped) escaped = false;
      else if (character === "\\") escaped = true;
      else if (character === '"') inString = false;
      continue;
    }
    if (character === '"') {
      inString = true;
      normalized += character;
      index += 1;
      continue;
    }
    if (character === "-" || /[0-9]/.test(character)) {
      const token = value
        .slice(index)
        .match(/^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/)?.[0];
      if (!token) throw new Error("Canonical JSON contains an invalid number.");
      const number = Number(token);
      if (!Number.isFinite(number)) {
        throw new Error("Canonical JSON contains a non-finite number.");
      }
      normalized += JSON.stringify(number);
      index += token.length;
      continue;
    }
    normalized += character;
    index += 1;
  }
  return normalized;
}

function packedManifestContentBytes(
  manifestText: string,
  claimedContentHash: string
): Buffer {
  const field = `"contentSha256":${JSON.stringify(claimedContentHash)}`;
  const fieldStart = manifestText.indexOf(field);
  if (
    fieldStart < 0 ||
    manifestText.indexOf(field, fieldStart + field.length) >= 0
  ) {
    throw new Error("Packed ternary content checksum field is ambiguous.");
  }
  let start = fieldStart;
  let end = fieldStart + field.length;
  if (manifestText[end] === ",") end += 1;
  else if (manifestText[start - 1] === ",") start -= 1;
  else throw new Error("Packed ternary content checksum field is malformed.");
  return Buffer.from(
    manifestText.slice(0, start) + manifestText.slice(end),
    "utf8"
  );
}

function safeSubstrateRelativePath(path: string): string {
  assertSafeArchivePath(path);
  if (
    path !== "manifest.json" &&
    !/^generations\/[a-f0-9]{64}\/manifest\.json$/.test(path) &&
    !/^blobs\/[a-f0-9]{64}\.(?:json|safetensors)$/.test(path)
  ) {
    throw new Error(`Neural substrate manifest contains an unsupported path: ${path}`);
  }
  return path;
}

interface SubstrateSnapshot {
  pointer: Record<string, unknown>;
  sources: StreamingZipSource[];
  relativePaths: Set<string>;
}

async function collectSubstrateSnapshot(
  engineDirectory: string,
  archivePrefix: string,
  engineMetadata: unknown
): Promise<SubstrateSnapshot | undefined> {
  if (!isRecord(engineMetadata) || !isRecord(engineMetadata.substrate)) return undefined;
  const embedded = engineMetadata.substrate.persistence;
  if (!isRecord(embedded)) return undefined;
  const store = join(engineDirectory, "substrate");
  // Engine metadata is the commit record. The root pointer may legitimately
  // name an orphaned newer generation after a process interruption, so export
  // follows the embedded record and synthesizes the matching portable pointer.
  if (
    embedded.format !== SUBSTRATE_STORE_FORMAT ||
    embedded.formatVersion !== 1 ||
    typeof embedded.generationManifest !== "string" ||
    typeof embedded.generationManifestSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(embedded.generationManifestSha256)
  ) {
    throw new Error("Neural substrate pointer is invalid.");
  }
  const generationRelative = safeSubstrateRelativePath(embedded.generationManifest);
  if (!generationRelative.startsWith("generations/")) {
    throw new Error("Neural substrate generation path is invalid.");
  }
  const generationPath = join(store, ...generationRelative.split("/"));
  const generationBytes = await readFile(generationPath);
  if (sha256(generationBytes) !== embedded.generationManifestSha256) {
    throw new Error("Neural substrate generation manifest checksum failed.");
  }
  let generation: unknown;
  try {
    generation = JSON.parse(generationBytes.toString("utf8"));
  } catch {
    throw new Error("Neural substrate generation manifest is invalid.");
  }
  if (
    !isRecord(generation) ||
    generation.format !== SUBSTRATE_STORE_FORMAT ||
    generation.formatVersion !== 1 ||
    !Array.isArray(generation.shards) ||
    typeof generation.contentSha256 !== "string" ||
    generation.contentSha256 !== embedded.activeGeneration ||
    embedded.contentSha256 !== embedded.activeGeneration ||
    !isRecord(embedded.counts) ||
    !isRecord(generation.counts) ||
    embedded.shardCount !== generation.shards.length ||
    canonicalJson(embedded.counts) !== canonicalJson(generation.counts)
  ) {
    throw new Error("Neural substrate generation manifest is incompatible.");
  }
  const content = { ...generation };
  delete content.contentSha256;
  if (sha256(canonicalJson(content)) !== generation.contentSha256) {
    throw new Error("Neural substrate generation content checksum failed.");
  }

  const declared = new Map<string, { sha256: string; bytes: number }>();
  for (const shard of generation.shards) {
    if (
      !isRecord(shard) ||
      !["neurons", "assemblies", "synapses"].includes(String(shard.kind)) ||
      typeof shard.bucket !== "string" ||
      !/^[a-f0-9]$/.test(shard.bucket) ||
      typeof shard.part !== "number" ||
      !Number.isSafeInteger(shard.part) ||
      shard.part < 0 ||
      typeof shard.count !== "number" ||
      !Number.isSafeInteger(shard.count) ||
      shard.count < 0
    ) {
      throw new Error("Neural substrate generation contains an invalid shard record.");
    }
    for (const key of ["records", "tensors"] as const) {
      const descriptor = shard[key];
      if (key === "tensors" && descriptor === null) continue;
      if (
        !isRecord(descriptor) ||
        typeof descriptor.path !== "string" ||
        typeof descriptor.sha256 !== "string" ||
        !/^[a-f0-9]{64}$/.test(descriptor.sha256) ||
        typeof descriptor.bytes !== "number" ||
        !Number.isSafeInteger(descriptor.bytes) ||
        descriptor.bytes < 0
      ) {
        throw new Error("Neural substrate generation contains an invalid blob descriptor.");
      }
      const relative = safeSubstrateRelativePath(descriptor.path);
      if (
        !relative.startsWith("blobs/") ||
        basename(relative).split(".")[0] !== descriptor.sha256
      ) {
        throw new Error("Neural substrate blob is not content-addressed by its checksum.");
      }
      const prior = declared.get(relative);
      const normalized = {
        sha256: descriptor.sha256,
        bytes: descriptor.bytes
      };
      if (prior && canonicalJson(prior) !== canonicalJson(normalized)) {
        throw new Error("Neural substrate generation contains conflicting blob descriptors.");
      }
      declared.set(relative, normalized);
    }
  }
  const sources: StreamingZipSource[] = [
    {
      name: `${archivePrefix}/manifest.json`,
      contents: Buffer.from(canonicalJson(embedded), "utf8")
    },
    {
      name: `${archivePrefix}/${generationRelative}`,
      sourcePath: generationPath
    }
  ];
  const relativePaths = new Set<string>(["manifest.json", generationRelative]);
  for (const [relative, descriptor] of [...declared].sort(([left], [right]) =>
    left.localeCompare(right)
  )) {
    const sourcePath = join(store, ...relative.split("/"));
    const info = await lstat(sourcePath);
    if (
      !info.isFile() ||
      info.isSymbolicLink() ||
      info.size !== descriptor.bytes ||
      (await streamFileSha256(sourcePath)) !== descriptor.sha256
    ) {
      throw new Error(`Neural substrate blob checksum failed: ${relative}`);
    }
    if (relative.endsWith(".safetensors")) {
      await assertSafeTensorsFile(sourcePath, `substrate shard ${relative}`);
    }
    sources.push({ name: `${archivePrefix}/${relative}`, sourcePath });
    relativePaths.add(relative);
  }
  return { pointer: embedded, sources, relativePaths };
}

async function validateExtractedSubstrateSnapshot(
  archive: ExtractedZipArchive,
  archivePrefix: string,
  engineMetadata: unknown
): Promise<Set<string>> {
  const matching = [...archive.entries.keys()].filter((name) =>
    name.startsWith(`${archivePrefix}/`)
  );
  if (!isRecord(engineMetadata) || !isRecord(engineMetadata.substrate)) {
    if (matching.length > 0) throw new Error("The bundle contains undeclared substrate shards.");
    return new Set();
  }
  const embedded = engineMetadata.substrate.persistence;
  if (!isRecord(embedded)) {
    if (matching.length > 0) throw new Error("The bundle contains undeclared substrate shards.");
    return new Set();
  }
  const pointerEntry = archive.entries.get(`${archivePrefix}/manifest.json`);
  if (!pointerEntry) throw new Error("The bundle is missing its substrate pointer.");
  const pointer = JSON.parse(await readFile(pointerEntry.path, "utf8")) as unknown;
  if (!isRecord(pointer) || canonicalJson(pointer) !== canonicalJson(embedded)) {
    throw new Error("The bundled substrate pointer does not match engine state.");
  }
  if (
    embedded.format !== SUBSTRATE_STORE_FORMAT ||
    embedded.formatVersion !== 1 ||
    typeof embedded.generationManifest !== "string" ||
    typeof embedded.generationManifestSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(embedded.generationManifestSha256) ||
    typeof embedded.activeGeneration !== "string" ||
    !/^[a-f0-9]{64}$/.test(embedded.activeGeneration)
  ) {
    throw new Error("The bundled substrate pointer is invalid.");
  }
  const generationRelative = safeSubstrateRelativePath(
    String(embedded.generationManifest ?? "")
  );
  const generationEntry = archive.entries.get(`${archivePrefix}/${generationRelative}`);
  if (
    !generationEntry ||
    (await streamFileSha256(generationEntry.path)) !==
      embedded.generationManifestSha256
  ) {
    throw new Error("The bundled substrate generation manifest checksum failed.");
  }
  const generation = JSON.parse(await readFile(generationEntry.path, "utf8")) as unknown;
  if (
    !isRecord(generation) ||
    generation.format !== SUBSTRATE_STORE_FORMAT ||
    generation.formatVersion !== 1 ||
    !Array.isArray(generation.shards) ||
    generation.contentSha256 !== embedded.activeGeneration ||
    embedded.contentSha256 !== embedded.activeGeneration ||
    !isRecord(embedded.counts) ||
    !isRecord(generation.counts) ||
    embedded.shardCount !== generation.shards.length ||
    canonicalJson(embedded.counts) !== canonicalJson(generation.counts)
  ) {
    throw new Error("The bundled substrate generation manifest is invalid.");
  }
  const generationBody = { ...generation };
  delete generationBody.contentSha256;
  if (sha256(canonicalJson(generationBody)) !== generation.contentSha256) {
    throw new Error("The bundled substrate generation content checksum failed.");
  }
  const declared = new Map<string, { sha256: string; bytes: number }>();
  for (const shard of generation.shards) {
    if (
      !isRecord(shard) ||
      !["neurons", "assemblies", "synapses"].includes(String(shard.kind)) ||
      typeof shard.bucket !== "string" ||
      !/^[a-f0-9]$/.test(shard.bucket) ||
      typeof shard.part !== "number" ||
      !Number.isSafeInteger(shard.part) ||
      shard.part < 0 ||
      typeof shard.count !== "number" ||
      !Number.isSafeInteger(shard.count) ||
      shard.count < 0
    ) {
      throw new Error("The bundled substrate shard table is invalid.");
    }
    for (const key of ["records", "tensors"] as const) {
      const descriptor = shard[key];
      if (key === "tensors" && descriptor === null) continue;
      if (
        !isRecord(descriptor) ||
        typeof descriptor.path !== "string" ||
        typeof descriptor.sha256 !== "string" ||
        !/^[a-f0-9]{64}$/.test(descriptor.sha256) ||
        typeof descriptor.bytes !== "number" ||
        !Number.isSafeInteger(descriptor.bytes) ||
        descriptor.bytes < 0
      ) {
        throw new Error("The bundled substrate blob descriptor is invalid.");
      }
      const relative = safeSubstrateRelativePath(descriptor.path);
      if (
        !relative.startsWith("blobs/") ||
        basename(relative).split(".")[0] !== descriptor.sha256
      ) {
        throw new Error("The bundled substrate blob is not content-addressed.");
      }
      const normalized = {
        sha256: descriptor.sha256,
        bytes: descriptor.bytes
      };
      const prior = declared.get(relative);
      if (prior && canonicalJson(prior) !== canonicalJson(normalized)) {
        throw new Error("The bundled substrate contains conflicting blob descriptors.");
      }
      declared.set(relative, normalized);
    }
  }
  const expected = new Set(["manifest.json", generationRelative, ...declared.keys()]);
  for (const relative of expected) {
    const entry = archive.entries.get(`${archivePrefix}/${relative}`);
    if (!entry) throw new Error(`The bundle is missing substrate file ${relative}.`);
    if (relative.startsWith("blobs/")) {
      const descriptor = declared.get(relative)!;
      if (
        entry.uncompressedBytes !== descriptor.bytes ||
        (await streamFileSha256(entry.path)) !== descriptor.sha256
      ) {
        throw new Error(`The bundled substrate blob checksum failed: ${relative}`);
      }
      if (relative.endsWith(".safetensors")) {
        await assertSafeTensorsFile(
          entry.path,
          `substrate shard ${relative}`
        );
      }
    }
  }
  if (
    matching.some((name) => !expected.has(name.slice(`${archivePrefix}/`.length)))
  ) {
    throw new Error("The bundle contains an unlisted substrate shard file.");
  }
  return expected;
}

function portableEngineState(
  contents: Buffer,
  includePrivateSources: boolean,
  redactions: RedactionCounter
): Uint8Array {
  let state: unknown;
  try {
    state = JSON.parse(contents.toString("utf8"));
  } catch {
    throw new Error("The Python engine metadata is invalid.");
  }
  if (!isRecord(state)) throw new Error("The Python engine metadata is invalid.");
  if (state.release_format !== STABLE_RELEASE_FORMAT) {
    throw new Error(
      "Incompatible OmniCortex beta engine; stable v1 bundles require stable neural state."
    );
  }
  if (!includePrivateSources && Array.isArray(state.training_sources)) {
    state.training_sources = state.training_sources.map((source) => {
      if (!isRecord(source)) return source;
      const sanitized = { ...source };
      delete sanitized.raw_text;
      sanitized.raw_text_retained = false;
      return sanitized;
    });
  }
  return strToU8(JSON.stringify(redactPortableValue(state, redactions), null, 2));
}

function requireSafeId(id: string, label = "brain id"): string {
  if (!SAFE_ID.test(id)) {
    throw new Error(`Invalid ${label}.`);
  }
  return id;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function boundNumber(value: unknown, fallback: number, minimum: number, maximum: number): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(minimum, Math.min(maximum, value))
    : fallback;
}

function normalizeConfig(value: unknown): BrainConfig {
  const config = isRecord(value) ? value : {};
  const merged = { ...DEFAULT_CONFIG } as BrainConfig;
  for (const key of Object.keys(DEFAULT_CONFIG)) {
    if (config[key] !== undefined) {
      (merged as unknown as Record<string, unknown>)[key] = config[key];
    }
  }
  merged.runtime = "adaptive-core";
  if (
    !["whole-brain", "ternary", "neuromorphic", "liquid", "symbolic", "custom"].includes(
      merged.preset
    )
  ) {
    merged.preset = "whole-brain";
  }
  if (!["summary", "standard", "research"].includes(merged.traceDetail)) {
    merged.traceDetail = "standard";
  }
  if (
    !["human-consolidation", "total-recall", "synapses-only"].includes(
      merged.memoryRecipe
    )
  ) {
    merged.memoryRecipe = "human-consolidation";
  }
  for (const key of [
    "onlineLearning",
    "extendedWorkingMemory",
    "recursiveImprovement",
    "idleCognition",
    "retainSourceText"
  ] as const) {
    if (typeof merged[key] !== "boolean") merged[key] = DEFAULT_CONFIG[key];
  }
  merged.name =
    typeof merged.name === "string" && merged.name.trim()
      ? merged.name.trim().slice(0, 120)
      : DEFAULT_CONFIG.name;
  merged.description =
    typeof merged.description === "string"
      ? merged.description.replace(/\0/g, "").slice(0, 4_000)
      : DEFAULT_CONFIG.description;
  merged.workingMemorySlots =
    typeof merged.workingMemorySlots === "number" &&
    Number.isSafeInteger(merged.workingMemorySlots) &&
    merged.workingMemorySlots > 0
      ? merged.workingMemorySlots
      : DEFAULT_CONFIG.workingMemorySlots;
  merged.learningRate = boundNumber(merged.learningRate, DEFAULT_CONFIG.learningRate, 0, 1);
  if (merged.memoryRecipe === "synapses-only") merged.retainSourceText = false;
  if (merged.memoryRecipe === "total-recall") merged.retainSourceText = true;
  return merged;
}

function normalizeBrain(value: unknown): BrainDocument {
  if (!isRecord(value)) throw new Error("The bundle does not contain a brain document.");
  if (value.releaseFormat !== STABLE_RELEASE_FORMAT) {
    throw new Error(
      "Incompatible Omni AGI Studio beta brain; create or import a stable v1 brain."
    );
  }
  const id = requireSafeId(String(value.id ?? ""));
  const now = new Date().toISOString();
  const lineageValue = isRecord(value.lineage) ? value.lineage : {};
  const countersValue = isRecord(value.counters) ? value.counters : {};
  const liquidValue = isRecord(value.liquidState) ? value.liquidState : {};

  const brain: BrainDocument = {
    schemaVersion: BRAIN_SCHEMA_VERSION,
    releaseFormat: STABLE_RELEASE_FORMAT,
    id,
    name: typeof value.name === "string" ? value.name.trim().slice(0, 120) || "Imported mind" : "Imported mind",
    createdAt: typeof value.createdAt === "string" ? value.createdAt : now,
    updatedAt: typeof value.updatedAt === "string" ? value.updatedAt : now,
    lineage: {
      parentId:
        typeof lineageValue.parentId === "string" && SAFE_ID.test(lineageValue.parentId)
          ? lineageValue.parentId
          : undefined,
      rootId:
        typeof lineageValue.rootId === "string" && SAFE_ID.test(lineageValue.rootId)
          ? lineageValue.rootId
          : id,
      generation: Math.max(0, Math.round(boundNumber(lineageValue.generation, 0, 0, 1_000_000)))
    },
    config: normalizeConfig(value.config),
    concepts: isRecord(value.concepts) ? (value.concepts as BrainDocument["concepts"]) : {},
    synapses: isRecord(value.synapses) ? (value.synapses as BrainDocument["synapses"]) : {},
    ideas: Array.isArray(value.ideas) ? (value.ideas as BrainDocument["ideas"]) : [],
    workingMemory: Array.isArray(value.workingMemory)
      ? (value.workingMemory as BrainDocument["workingMemory"])
      : [],
    liquidState: {
      values: Array.isArray(liquidValue.values)
        ? liquidValue.values.filter((entry): entry is number => typeof entry === "number" && Number.isFinite(entry))
        : Array.from({ length: 16 }, () => 0),
      timeConstants: Array.isArray(liquidValue.timeConstants)
        ? liquidValue.timeConstants.filter(
            (entry): entry is number => typeof entry === "number" && Number.isFinite(entry)
          )
        : Array.from({ length: 16 }, (_, index) => 0.25 + index * 0.05),
      lastUpdatedAt:
        typeof liquidValue.lastUpdatedAt === "string" ? liquidValue.lastUpdatedAt : now
    },
    messages: Array.isArray(value.messages) ? (value.messages as BrainDocument["messages"]) : [],
    traces: Array.isArray(value.traces) ? (value.traces as BrainDocument["traces"]) : [],
    trainingSources: Array.isArray(value.trainingSources)
      ? (value.trainingSources as BrainDocument["trainingSources"])
      : [],
    counters: {
      plasticityEvents: Math.max(
        0,
        Math.round(boundNumber(countersValue.plasticityEvents, 0, 0, Number.MAX_SAFE_INTEGER))
      ),
      inferenceCount: Math.max(
        0,
        Math.round(boundNumber(countersValue.inferenceCount, 0, 0, Number.MAX_SAFE_INTEGER))
      ),
      consolidationCycles: Math.max(
        0,
        Math.round(boundNumber(countersValue.consolidationCycles, 0, 0, Number.MAX_SAFE_INTEGER))
      )
    },
    toolPermissions: Array.isArray(value.toolPermissions)
      ? (value.toolPermissions as ToolPermissionRecord[])
      : clone(DEFAULT_TOOL_PERMISSIONS),
    journal: Array.isArray(value.journal) ? (value.journal as BrainDocument["journal"]) : [],
    originChecksum: typeof value.originChecksum === "string" ? value.originChecksum : undefined
  };
  brain.config.name = brain.name;
  return brain;
}

export function brainMetrics(brain: BrainDocument): BrainMetrics {
  const synapses = Object.values(brain.synapses);
  const estimatedBytes = Buffer.byteLength(JSON.stringify(brain), "utf8");
  return {
    concepts: Object.keys(brain.concepts).length,
    synapses: synapses.length,
    activeSynapses: synapses.filter((synapse) => synapse.effectiveWeight !== 0).length,
    ideas: brain.ideas.length,
    messages: brain.messages.length,
    trainingSources: brain.trainingSources.length,
    averageStability:
      synapses.length === 0
        ? 0
        : synapses.reduce((sum, synapse) => sum + synapse.stability, 0) / synapses.length,
    plasticityEvents: brain.counters.plasticityEvents,
    inferenceCount: brain.counters.inferenceCount,
    estimatedBytes
  };
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function atomicWrite(path: string, contents: string | Buffer): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${randomUUID()}.next`;
  const handle = await open(temporary, "wx", 0o600);
  try {
    await handle.writeFile(contents);
    await handle.sync();
  } finally {
    await handle.close();
  }

  try {
    await rename(temporary, path);
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code !== "EEXIST" && code !== "EPERM") {
      await rm(temporary, { force: true });
      throw error;
    }

    const backup = `${path}.bak`;
    if (await pathExists(path)) await rename(path, backup);
    try {
      await rename(temporary, path);
      await rm(backup, { force: true });
    } catch (replacementError) {
      if (await pathExists(backup)) await rename(backup, path);
      await rm(temporary, { force: true });
      throw replacementError;
    }
  }
}

async function inspectPackedTernaryDirectory(
  directory: string,
  label: string,
  overrides: Map<string, string> = new Map()
): Promise<StreamingPackedTernaryDirectory> {
  const directoryEntries = await readdir(directory, { withFileTypes: true });
  if (directoryEntries.some((entry) => !entry.isFile())) {
    throw new Error(`${label} packed ternary directory contains a non-file entry.`);
  }
  const inventory = new Set(directoryEntries.map((entry) => entry.name));
  const filePath = (name: string): string => {
    if (!inventory.has(name)) {
      throw new Error(`${label} packed ternary directory is missing ${name}.`);
    }
    return overrides.get(name) ?? join(directory, name);
  };
  const manifestBytes = await readFile(filePath("manifest.json"));
  const expectedManifestHash = (
    await readFile(filePath("manifest.sha256"), "ascii")
  ).trim();
  if (
    !/^[a-f0-9]{64}$/.test(expectedManifestHash) ||
    sha256(manifestBytes) !== expectedManifestHash
  ) {
    throw new Error(`${label} packed ternary manifest checksum failed.`);
  }
  let parsed: unknown;
  const manifestText = manifestBytes.toString("utf8");
  try {
    parsed = JSON.parse(manifestText);
  } catch {
    throw new Error(`${label} packed ternary manifest is invalid.`);
  }
  if (
    !isRecord(parsed) ||
    parsed.format !== "omni-packed-ternary" ||
    parsed.formatVersion !== 1 ||
    parsed.architecture !== "OmniCortex" ||
    !isRecord(parsed.encoding) ||
    parsed.encoding.bitsPerValue !== 2 ||
    parsed.encoding.byteOrder !== "four-values-lsb-first" ||
    parsed.encoding.reservedCode !== 3 ||
    parsed.encoding.paddingValue !== 0 ||
    !isRecord(parsed.encoding.codes) ||
    parsed.encoding.codes["-1"] !== 0 ||
    parsed.encoding.codes["0"] !== 1 ||
    parsed.encoding.codes["+1"] !== 2 ||
    !isRecord(parsed.coverage) ||
    parsed.coverage.complete !== true ||
    !Array.isArray(parsed.coverage.eligibleTensorNames) ||
    !Array.isArray(parsed.tensors) ||
    !Array.isArray(parsed.shards)
  ) {
    throw new Error(`${label} packed ternary manifest is incompatible.`);
  }
  // Python deliberately preserves float identity (for example `1.0`) while
  // JavaScript JSON.parse represents it as the number `1`. Normalize only
  // numeric lexemes before comparing canonical structure so key ordering,
  // whitespace, string escaping, and duplicate keys remain strictly checked.
  if (normalizeCanonicalJsonNumbers(manifestText) !== canonicalJson(parsed)) {
    throw new Error(`${label} packed ternary manifest is not canonical.`);
  }
  const claimedContentHash = parsed.contentSha256;
  if (
    typeof claimedContentHash !== "string" ||
    !/^[a-f0-9]{64}$/.test(claimedContentHash) ||
    sha256(packedManifestContentBytes(manifestText, claimedContentHash)) !==
      claimedContentHash
  ) {
    throw new Error(`${label} packed ternary content checksum failed.`);
  }
  const expectedNames = parsed.coverage.eligibleTensorNames;
  if (
    expectedNames.some((name) => typeof name !== "string" || !name) ||
    new Set(expectedNames).size !== expectedNames.length ||
    parsed.coverage.eligibleTensorCount !== expectedNames.length ||
    parsed.tensors.length !== expectedNames.length
  ) {
    throw new Error(`${label} packed ternary coverage contract is invalid.`);
  }
  const shardTable = new Map<
    string,
    { byteLength: number; sha256: string; path: string }
  >();
  for (const descriptor of parsed.shards) {
    if (
      !isRecord(descriptor) ||
      typeof descriptor.file !== "string" ||
      !/^ternary-[0-9]{5,}-[a-f0-9]{16}\.bin$/.test(descriptor.file) ||
      typeof descriptor.byteLength !== "number" ||
      !Number.isSafeInteger(descriptor.byteLength) ||
      descriptor.byteLength < 0 ||
      typeof descriptor.sha256 !== "string" ||
      !/^[a-f0-9]{64}$/.test(descriptor.sha256) ||
      shardTable.has(descriptor.file)
    ) {
      throw new Error(`${label} packed ternary shard table is invalid.`);
    }
    const path = filePath(descriptor.file);
    const info = await lstat(path);
    if (
      !info.isFile() ||
      info.isSymbolicLink() ||
      info.size !== descriptor.byteLength ||
      (await streamFileSha256(path)) !== descriptor.sha256
    ) {
      throw new Error(`${label} packed ternary shard checksum failed.`);
    }
    shardTable.set(descriptor.file, {
      byteLength: descriptor.byteLength,
      sha256: descriptor.sha256,
      path
    });
  }
  const tensorNames: string[] = [];
  const usedShards = new Set<string>();
  for (const tensor of parsed.tensors) {
    if (
      !isRecord(tensor) ||
      typeof tensor.name !== "string" ||
      typeof tensor.shard !== "string" ||
      !shardTable.has(tensor.shard) ||
      usedShards.has(tensor.shard) ||
      !["projection", "dynamic-synapse"].includes(String(tensor.kind)) ||
      tensor.dtype !== "int8" ||
      !Array.isArray(tensor.shape) ||
      tensor.shape.some(
        (dimension) =>
          typeof dimension !== "number" ||
          !Number.isSafeInteger(dimension) ||
          dimension < 0
      ) ||
      typeof tensor.numel !== "number" ||
      !Number.isSafeInteger(tensor.numel) ||
      tensor.numel < 0 ||
      tensor.byteOffset !== 0 ||
      typeof tensor.byteLength !== "number" ||
      !Number.isSafeInteger(tensor.byteLength) ||
      tensor.byteLength < 0 ||
      typeof tensor.scale !== "number" ||
      !Number.isFinite(tensor.scale) ||
      tensor.scale <= 0 ||
      typeof tensor.sourceDtype !== "string" ||
      typeof tensor.packedSha256 !== "string" ||
      !/^[a-f0-9]{64}$/.test(tensor.packedSha256) ||
      typeof tensor.tensorSha256 !== "string" ||
      !/^[a-f0-9]{64}$/.test(tensor.tensorSha256)
    ) {
      throw new Error(`${label} packed ternary tensor table is invalid.`);
    }
    const shapeProduct = tensor.shape.reduce(
      (total, dimension) => total * (dimension as number),
      1
    );
    const shard = shardTable.get(tensor.shard)!;
    const expectedByteLength = Math.ceil(tensor.numel / 4);
    if (
      !Number.isSafeInteger(shapeProduct) ||
      shapeProduct !== tensor.numel ||
      tensor.byteLength !== expectedByteLength ||
      shard.byteLength !== expectedByteLength ||
      tensor.packedSha256 !== shard.sha256
    ) {
      throw new Error(`${label} packed ternary tensor shape or length is invalid.`);
    }
    const decodedDigest = createHash("sha256");
    decodedDigest.update(
      Buffer.from(JSON.stringify({ dtype: "int8", shape: tensor.shape }), "utf8")
    );
    decodedDigest.update(Buffer.from([0]));
    let decoded = 0;
    for await (const chunk of createReadStream(shard.path)) {
      const value = chunk as Buffer;
      const decodedChunk = Buffer.allocUnsafe(
        Math.min(tensor.numel - decoded, value.byteLength * 4)
      );
      let chunkDecoded = 0;
      for (let byteIndex = 0; byteIndex < value.byteLength; byteIndex += 1) {
        for (let slot = 0; slot < 4; slot += 1) {
          const code = (value[byteIndex]! >> (slot * 2)) & 0x03;
          if (decoded + chunkDecoded >= tensor.numel) {
            if (code !== 1) {
              throw new Error(`${label} packed ternary padding is non-canonical.`);
            }
          } else {
            if (code === 3) {
              throw new Error(`${label} packed ternary data uses the reserved code.`);
            }
            decodedChunk[chunkDecoded] = code === 0 ? 0xff : code === 1 ? 0 : 1;
            chunkDecoded += 1;
          }
        }
      }
      decoded += chunkDecoded;
      decodedDigest.update(decodedChunk.subarray(0, chunkDecoded));
    }
    if (decoded !== tensor.numel || decodedDigest.digest("hex") !== tensor.tensorSha256) {
      throw new Error(`${label} packed ternary decoded tensor checksum failed.`);
    }
    tensorNames.push(tensor.name);
    usedShards.add(tensor.shard);
  }
  if (
    tensorNames.length !== expectedNames.length ||
    tensorNames.some((name, index) => name !== expectedNames[index]) ||
    usedShards.size !== shardTable.size
  ) {
    throw new Error(`${label} packed ternary coverage is incomplete.`);
  }
  const allowed = new Set(["manifest.json", "manifest.sha256", ...shardTable.keys()]);
  if ([...inventory].some((name) => !allowed.has(name))) {
    throw new Error(`${label} packed ternary directory contains an unlisted file.`);
  }
  return {
    manifestSha256: expectedManifestHash,
    tensorCount: expectedNames.length,
    files: new Map([...allowed].map((name) => [name, filePath(name)]))
  };
}

export function resolveBrainDataRoot(
  userDataPath: string,
  override = process.env.OMNI_AGI_DATA_DIR
): string {
  const windowsLocal =
    process.platform === "win32" && process.env.LOCALAPPDATA
      ? join(resolve(process.env.LOCALAPPDATA), "OmniAGI")
      : undefined;
  const base = override?.trim()
    ? resolve(override.trim())
    : windowsLocal ?? resolve(userDataPath);
  return join(base, "brains");
}

export class BrainRepository {
  readonly root: string;

  constructor(root: string) {
    this.root = resolve(root);
  }

  async initialize(): Promise<void> {
    await Promise.all([
      mkdir(this.root, { recursive: true }),
      mkdir(join(this.root, ".trash"), { recursive: true }),
      mkdir(join(this.root, ".blobs"), { recursive: true })
    ]);
  }

  async betaReviewComplete(): Promise<boolean> {
    try {
      const value = JSON.parse(
        await readFile(join(this.root, BETA_REVIEW_FILE), "utf8")
      ) as unknown;
      return (
        isRecord(value) &&
        value.releaseFormat === STABLE_RELEASE_FORMAT &&
        typeof value.reviewedAt === "string"
      );
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
      return false;
    }
  }

  async enumerateManagedBetaBrains(): Promise<ManagedBetaBrain[]> {
    await this.initialize();
    const entries = await readdir(this.root, { withFileTypes: true });
    const candidates: ManagedBetaBrain[] = [];
    for (const entry of entries) {
      if (!entry.isDirectory() || !SAFE_ID.test(entry.name)) continue;
      const directory = join(this.root, entry.name);
      const documentPath = join(directory, "brain.json");
      if (!(await pathExists(documentPath))) continue;
      let document: Record<string, unknown> | undefined;
      try {
        const parsed = JSON.parse(await readFile(documentPath, "utf8")) as unknown;
        document = isRecord(parsed) ? parsed : undefined;
      } catch {
        document = undefined;
      }
      const name =
        typeof document?.name === "string" && document.name.trim()
          ? document.name.trim().slice(0, 120)
          : entry.name;
      if (!document) {
        candidates.push({
          id: entry.name,
          name,
          path: directory,
          reason: "invalid-document"
        });
        continue;
      }
      if (document.releaseFormat !== STABLE_RELEASE_FORMAT) {
        candidates.push({
          id: entry.name,
          name,
          path: directory,
          reason: "beta-document"
        });
        continue;
      }
      const enginePath = join(directory, "engine", "brain.json");
      if (!(await pathExists(enginePath))) continue;
      try {
        const engineState = JSON.parse(await readFile(enginePath, "utf8")) as unknown;
        if (
          !isRecord(engineState) ||
          engineState.release_format !== STABLE_RELEASE_FORMAT
        ) {
          candidates.push({
            id: entry.name,
            name,
            path: directory,
            reason: "beta-engine"
          });
        }
      } catch {
        candidates.push({
          id: entry.name,
          name,
          path: directory,
          reason: "beta-engine"
        });
      }
    }
    return candidates.sort((left, right) => left.id.localeCompare(right.id));
  }

  async deleteManagedBetaBrains(
    ids: string[],
    explicitlyConfirmed: boolean
  ): Promise<string[]> {
    if (!explicitlyConfirmed) {
      throw new Error("Permanent beta deletion requires explicit confirmation.");
    }
    const requested = [...new Set(ids.map((id) => requireSafeId(id)))];
    const candidates = new Map(
      (await this.enumerateManagedBetaBrains()).map((candidate) => [
        candidate.id,
        candidate
      ])
    );
    const rootPath = await realpath(this.root);
    const deleted: string[] = [];
    for (const id of requested) {
      const candidate = candidates.get(id);
      if (!candidate) {
        throw new Error(`Managed beta brain "${id}" is no longer eligible for deletion.`);
      }
      const targetPath = await realpath(candidate.path);
      if (dirname(targetPath) !== rootPath || basename(targetPath) !== id) {
        throw new Error("Managed beta deletion escaped the app data root.");
      }
      const info = await stat(targetPath);
      if (!info.isDirectory()) {
        throw new Error(`Managed beta brain "${id}" is not a directory.`);
      }
      await rm(targetPath, { recursive: true, force: false });
      deleted.push(id);
    }
    return deleted;
  }

  async completeBetaReview(
    disposition: "kept" | "deleted" | "none",
    ids: string[]
  ): Promise<void> {
    await this.initialize();
    await atomicWrite(
      join(this.root, BETA_REVIEW_FILE),
      JSON.stringify(
        {
          releaseFormat: STABLE_RELEASE_FORMAT,
          reviewedAt: new Date().toISOString(),
          disposition,
          managedBrainIds: [...new Set(ids.map((id) => requireSafeId(id)))]
        },
        null,
        2
      )
    );
  }

  async storeBlob(contents: Buffer): Promise<string> {
    await this.initialize();
    const hash = sha256(contents);
    const destination = join(this.root, ".blobs", hash);
    if (!(await pathExists(destination))) {
      try {
        await writeFile(destination, contents, { flag: "wx", mode: 0o600 });
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
      }
    }
    return hash;
  }

  async getBlob(hash: string): Promise<Buffer> {
    if (!/^[a-f0-9]{64}$/.test(hash)) throw new Error("Invalid content-addressed blob hash.");
    const contents = await readFile(join(this.root, ".blobs", hash));
    if (sha256(contents) !== hash) throw new Error("Content-addressed blob checksum failed.");
    return contents;
  }

  async storeFileAsBlob(path: string): Promise<string> {
    await this.initialize();
    const temporary = join(this.root, ".blobs", `.incoming-${randomUUID()}`);
    try {
      // App-managed source files remain mutable. Copy into the immutable blob
      // store; hard-linking the source here would let a later in-place write
      // corrupt every snapshot and bundle reference sharing that inode.
      await copyFile(path, temporary);
      const hash = await fileSha256(temporary);
      const destination = join(this.root, ".blobs", hash);
      if (await pathExists(destination)) await rm(temporary, { force: true });
      else {
        try {
          await rename(temporary, destination);
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
          await rm(temporary, { force: true });
        }
      }
      return hash;
    } catch (error) {
      await rm(temporary, { force: true });
      throw error;
    }
  }

  async linkBlobTo(hash: string, destination: string): Promise<void> {
    const source = join(this.root, ".blobs", hash);
    if (!/^[a-f0-9]{64}$/.test(hash) || (await fileSha256(source)) !== hash) {
      throw new Error("Content-addressed blob checksum failed.");
    }
    const sourceInfo = await stat(source);
    await mkdir(dirname(destination), { recursive: true });
    const temporary = `${destination}.${randomUUID()}.next`;
    try {
      await link(source, temporary);
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (!["EXDEV", "EPERM", "EACCES", "ENOTSUP"].includes(code ?? "")) throw error;
      await copyFile(source, temporary);
    }
    if ((await stat(temporary)).size !== sourceInfo.size) {
      await rm(temporary, { force: true });
      throw new Error("Copy-on-write blob materialization failed.");
    }
    try {
      await rename(temporary, destination);
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (code !== "EEXIST" && code !== "EPERM") {
        await rm(temporary, { force: true });
        throw error;
      }
      const backup = `${destination}.bak`;
      if (await pathExists(destination)) await rename(destination, backup);
      try {
        await rename(temporary, destination);
        await rm(backup, { force: true });
      } catch (replacementError) {
        if (await pathExists(backup)) await rename(backup, destination);
        throw replacementError;
      }
    }
  }

  private async copyPackedTernaryDirectory(
    source: string,
    destination: string
  ): Promise<StreamingPackedTernaryDirectory | undefined> {
    if (!(await pathExists(join(source, "manifest.json")))) return undefined;
    const packed = await inspectPackedTernaryDirectory(source, "Source");
    const temporary = `${destination}.${randomUUID()}.next`;
    const backup = `${destination}.${randomUUID()}.bak`;
    await mkdir(temporary, { recursive: true });
    try {
      for (const [name, sourcePath] of packed.files) {
        const hash = await this.storeFileAsBlob(sourcePath);
        await this.linkBlobTo(hash, join(temporary, name));
      }
      if (await pathExists(destination)) await rename(destination, backup);
      try {
        await rename(temporary, destination);
        await rm(backup, { recursive: true, force: true });
      } catch (error) {
        if (await pathExists(backup)) await rename(backup, destination);
        throw error;
      }
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
    await inspectPackedTernaryDirectory(destination, "Copied");
    return packed;
  }

  private async copySubstrateSnapshot(
    sourceEngine: string,
    destinationEngine: string,
    metadata: unknown
  ): Promise<string | undefined> {
    const prefix = "substrate/snapshot";
    const snapshot = await collectSubstrateSnapshot(
      sourceEngine,
      prefix,
      metadata
    );
    if (!snapshot) return undefined;
    const destination = join(destinationEngine, "substrate");
    const temporary = `${destination}.${randomUUID()}.next`;
    const backup = `${destination}.${randomUUID()}.bak`;
    await mkdir(temporary, { recursive: true });
    try {
      for (const source of snapshot.sources) {
        const relative = source.name.slice(`${prefix}/`.length);
        const hash = source.sourcePath
          ? await this.storeFileAsBlob(source.sourcePath)
          : await this.storeBlob(Buffer.from(source.contents!));
        await this.linkBlobTo(hash, join(temporary, ...relative.split("/")));
      }
      if (await pathExists(destination)) await rename(destination, backup);
      try {
        await rename(temporary, destination);
        await rm(backup, { recursive: true, force: true });
      } catch (error) {
        if (await pathExists(backup)) await rename(backup, destination);
        throw error;
      }
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
    await collectSubstrateSnapshot(destinationEngine, prefix, metadata);
    return sha256(canonicalJson(snapshot.pointer));
  }

  private async cloneEngineState(
    sourceBrainId: string,
    targetBrainId: string,
    targetName: string
  ): Promise<void> {
    const sourceEngine = join(this.brainDirectory(sourceBrainId), "engine");
    const metadataPath = join(sourceEngine, "brain.json");
    if (!(await pathExists(metadataPath))) return;
    const metadata = JSON.parse(await readFile(metadataPath, "utf8")) as unknown;
    if (!isRecord(metadata)) throw new Error("The source engine metadata is invalid.");
    metadata.brain_id = targetBrainId;
    metadata.name = targetName;
    if (isRecord(metadata.config)) metadata.config.name = targetName;
    const targetEngine = join(this.brainDirectory(targetBrainId), "engine");
    const targetOrigin = join(targetEngine, "origin");
    await Promise.all([
      mkdir(targetEngine, { recursive: true }),
      mkdir(targetOrigin, { recursive: true })
    ]);
    const tensors = [
      ["core.safetensors", "core.safetensors"],
      ["plasticity.safetensors", "plasticity.safetensors"]
    ] as const;
    for (const [sourceName, targetNameValue] of tensors) {
      const sourcePath = join(sourceEngine, sourceName);
      if (!(await pathExists(sourcePath))) continue;
      const hash = await this.storeFileAsBlob(sourcePath);
      await Promise.all([
        this.linkBlobTo(hash, join(targetEngine, targetNameValue)),
        this.linkBlobTo(hash, join(targetOrigin, targetNameValue))
      ]);
    }
    const packedSource = join(sourceEngine, "packed-ternary");
    // Wait for every concurrent materializer before the caller can remove a
    // failed clone. Promise.all would reject early while sibling operations
    // continued recreating paths underneath the cleanup, leaving an orphaned
    // partial brain after any validator failure.
    const materialized = await Promise.allSettled([
      this.copyPackedTernaryDirectory(
        packedSource,
        join(targetEngine, "packed-ternary")
      ),
      this.copyPackedTernaryDirectory(
        packedSource,
        join(targetOrigin, "packed-ternary")
      ),
      this.copySubstrateSnapshot(sourceEngine, targetEngine, metadata),
      this.copySubstrateSnapshot(sourceEngine, targetOrigin, metadata)
    ]);
    const failedMaterialization = materialized.find(
      (result): result is PromiseRejectedResult => result.status === "rejected"
    );
    if (failedMaterialization) throw failedMaterialization.reason;
    // Metadata is the commit record and therefore moves last.
    await Promise.all([
      atomicWrite(join(targetEngine, "brain.json"), JSON.stringify(metadata, null, 2)),
      atomicWrite(join(targetOrigin, "brain.json"), JSON.stringify(metadata, null, 2))
    ]);
  }

  brainDirectory(id: string): string {
    return join(this.root, requireSafeId(id));
  }

  private documentPath(id: string): string {
    return join(this.brainDirectory(id), "brain.json");
  }

  async create(config: BrainConfig): Promise<BrainDocument> {
    await this.initialize();
    const id = randomUUID();
    const now = new Date().toISOString();
    const normalizedConfig = normalizeConfig(config);
    const brain: BrainDocument = {
      schemaVersion: BRAIN_SCHEMA_VERSION,
      releaseFormat: STABLE_RELEASE_FORMAT,
      id,
      name: normalizedConfig.name,
      createdAt: now,
      updatedAt: now,
      lineage: { rootId: id, generation: 0 },
      config: normalizedConfig,
      concepts: {},
      synapses: {},
      ideas: [],
      workingMemory: [],
      liquidState: {
        values: Array.from({ length: 16 }, () => 0),
        timeConstants: Array.from({ length: 16 }, (_, index) => 0.25 + index * 0.05),
        lastUpdatedAt: now
      },
      messages: [],
      traces: [],
      trainingSources: [],
      counters: { plasticityEvents: 0, inferenceCount: 0, consolidationCycles: 0 },
      toolPermissions: clone(DEFAULT_TOOL_PERMISSIONS),
      journal: [
        {
          id: randomUUID(),
          createdAt: now,
          kind: "system",
          summary: "Immutable origin created."
        }
      ]
    };
    brain.originChecksum = sha256(JSON.stringify({ ...brain, originChecksum: undefined }));
    const directory = this.brainDirectory(id);
    await mkdir(join(directory, "snapshots"), { recursive: true });
    await atomicWrite(this.documentPath(id), JSON.stringify(brain, null, 2));
    await writeFile(join(directory, "origin.json"), JSON.stringify(brain, null, 2), {
      encoding: "utf8",
      flag: "wx",
      mode: 0o600
    });
    return clone(brain);
  }

  async get(id: string): Promise<BrainDocument> {
    const documentPath = this.documentPath(id);
    try {
      const raw = await readFile(documentPath, "utf8");
      return normalizeBrain(JSON.parse(raw));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      const backup = `${documentPath}.bak`;
      if (!(await pathExists(backup))) throw new Error(`Brain "${id}" was not found.`);
      const recovered = await readFile(backup, "utf8");
      const brain = normalizeBrain(JSON.parse(recovered));
      await atomicWrite(documentPath, JSON.stringify(brain, null, 2));
      return brain;
    }
  }

  async save(brain: BrainDocument, touch = true): Promise<BrainDocument> {
    const normalized = normalizeBrain(brain);
    if (touch) normalized.updatedAt = new Date().toISOString();
    normalized.name = normalized.config.name.trim() || normalized.name;
    await atomicWrite(this.documentPath(normalized.id), JSON.stringify(normalized, null, 2));
    return clone(normalized);
  }

  async list(): Promise<BrainSummary[]> {
    await this.initialize();
    const entries = await readdir(this.root, { withFileTypes: true });
    const summaries = await Promise.all(
      entries
        .filter((entry) => entry.isDirectory() && SAFE_ID.test(entry.name))
        .map(async (entry): Promise<BrainSummary | undefined> => {
          try {
            const brain = await this.get(entry.name);
            return {
              id: brain.id,
              name: brain.name,
              preset: brain.config.preset,
              runtime: brain.config.runtime,
              updatedAt: brain.updatedAt,
              concepts: Object.keys(brain.concepts).length,
              synapses: Object.keys(brain.synapses).length,
              generation: brain.lineage.generation
            };
          } catch {
            return undefined;
          }
        })
    );
    return summaries
      .filter((summary): summary is BrainSummary => summary !== undefined)
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
  }

  async updateConfig(id: string, config: BrainConfig): Promise<BrainDocument> {
    const brain = await this.get(id);
    brain.config = normalizeConfig(config);
    brain.name = brain.config.name;
    return this.save(brain);
  }

  async fork(id: string, name?: string): Promise<BrainDocument> {
    return this.copyOnWriteClone(id, name, "fork");
  }

  async duplicate(id: string, name?: string): Promise<BrainDocument> {
    return this.copyOnWriteClone(id, name, "duplicate");
  }

  private async copyOnWriteClone(
    id: string,
    name: string | undefined,
    operation: "fork" | "duplicate"
  ): Promise<BrainDocument> {
    const source = await this.get(id);
    const fork = clone(source);
    const now = new Date().toISOString();
    fork.id = randomUUID();
    fork.name =
      name?.trim().slice(0, 120) ||
      `${source.name}${operation === "duplicate" ? " copy" : " fork"}`;
    fork.config.name = fork.name;
    fork.createdAt = now;
    fork.updatedAt = now;
    fork.lineage = {
      parentId: source.id,
      rootId: source.lineage.rootId,
      generation: source.lineage.generation + 1
    };
    fork.journal = [
      ...(fork.journal ?? []),
      {
        id: randomUUID(),
        createdAt: now,
        kind: "fork",
        summary:
          operation === "duplicate"
            ? `Duplicated from ${source.name} with copy-on-write neural storage.`
            : `Forked from ${source.name}.`,
        detail: JSON.stringify({
          sourceBrainId: source.id,
          operation,
          copyOnWrite: true
        })
      }
    ];
    fork.originChecksum = undefined;
    fork.originChecksum = sha256(JSON.stringify(fork));
    const directory = this.brainDirectory(fork.id);
    try {
      await mkdir(join(directory, "snapshots"), { recursive: true });
      await atomicWrite(this.documentPath(fork.id), JSON.stringify(fork, null, 2));
      await writeFile(join(directory, "origin.json"), JSON.stringify(fork, null, 2), {
        encoding: "utf8",
        flag: "wx",
        mode: 0o600
      });
      await this.cloneEngineState(source.id, fork.id, fork.name);
      return clone(fork);
    } catch (error) {
      await rm(directory, { recursive: true, force: true });
      throw error;
    }
  }

  async remove(id: string): Promise<boolean> {
    const source = this.brainDirectory(id);
    if (!(await pathExists(source))) return false;
    const trashName = `${requireSafeId(id)}-${Date.now()}`;
    await rename(source, join(this.root, ".trash", trashName));
    return true;
  }

  async snapshot(id: string, label?: string): Promise<BrainSnapshotSummary> {
    const brain = await this.get(id);
    const snapshotId = randomUUID();
    const createdAt = new Date().toISOString();
    const document = JSON.stringify(brain, null, 2);
    const engineSource = join(this.brainDirectory(id), "engine");
    const engineSnapshot = join(this.brainDirectory(id), "snapshots", snapshotId, "engine");
    let engineChecksum: string | undefined;
    if (await pathExists(join(engineSource, "brain.json"))) {
      await mkdir(engineSnapshot, { recursive: true });
      const metadata = await readFile(join(engineSource, "brain.json"));
      const metadataValue = JSON.parse(metadata.toString("utf8")) as unknown;
      const hashes: string[] = [sha256(metadata)];
      for (const name of ["core.safetensors", "plasticity.safetensors"]) {
        const sourcePath = join(engineSource, name);
        if (!(await pathExists(sourcePath))) continue;
        const hash = await this.storeFileAsBlob(sourcePath);
        await this.linkBlobTo(hash, join(engineSnapshot, name));
        hashes.push(hash);
      }
      const packed = await this.copyPackedTernaryDirectory(
        join(engineSource, "packed-ternary"),
        join(engineSnapshot, "packed-ternary")
      );
      if (packed) hashes.push(packed.manifestSha256);
      const substrateHash = await this.copySubstrateSnapshot(
        engineSource,
        engineSnapshot,
        metadataValue
      );
      if (substrateHash) hashes.push(substrateHash);
      // Commit metadata after every referenced shard is durable.
      await atomicWrite(join(engineSnapshot, "brain.json"), metadata);
      engineChecksum = sha256(hashes.join(":"));
    }
    const summary: BrainSnapshotSummary = {
      id: snapshotId,
      brainId: brain.id,
      label: label?.trim().slice(0, 120) || `Snapshot ${createdAt}`,
      createdAt,
      checksum: sha256(document),
      metrics: brainMetrics(brain),
      engineChecksum
    };
    const base = join(this.brainDirectory(id), "snapshots", snapshotId);
    await writeFile(`${base}.json`, document, { encoding: "utf8", flag: "wx", mode: 0o600 });
    await writeFile(`${base}.meta.json`, JSON.stringify(summary, null, 2), {
      encoding: "utf8",
      flag: "wx",
      mode: 0o600
    });
    return summary;
  }

  async listSnapshots(id: string): Promise<BrainSnapshotSummary[]> {
    const directory = join(this.brainDirectory(id), "snapshots");
    await mkdir(directory, { recursive: true });
    const entries = await readdir(directory, { withFileTypes: true });
    const snapshots = await Promise.all(
      entries
        .filter((entry) => entry.isFile() && entry.name.endsWith(".meta.json"))
        .map(async (entry): Promise<BrainSnapshotSummary | undefined> => {
          try {
            const value = JSON.parse(await readFile(join(directory, entry.name), "utf8")) as unknown;
            if (!isRecord(value) || value.brainId !== id || typeof value.id !== "string") return undefined;
            return value as unknown as BrainSnapshotSummary;
          } catch {
            return undefined;
          }
        })
    );
    return snapshots
      .filter((snapshot): snapshot is BrainSnapshotSummary => snapshot !== undefined)
      .sort((left, right) => right.createdAt.localeCompare(left.createdAt));
  }

  async restoreSnapshot(id: string, snapshotId: string): Promise<BrainDocument> {
    requireSafeId(snapshotId, "snapshot id");
    const current = await this.get(id);
    const base = join(this.brainDirectory(id), "snapshots", snapshotId);
    const [document, metadata] = await Promise.all([
      readFile(`${base}.json`, "utf8"),
      readFile(`${base}.meta.json`, "utf8")
    ]);
    const summary = JSON.parse(metadata) as BrainSnapshotSummary;
    if (summary.brainId !== id || sha256(document) !== summary.checksum) {
      throw new Error("Snapshot checksum validation failed.");
    }
    const restored = normalizeBrain(JSON.parse(document));
    restored.id = id;
    restored.lineage = current.lineage;
    restored.createdAt = current.createdAt;
    restored.journal = [
      ...(restored.journal ?? []),
      {
        id: randomUUID(),
        createdAt: new Date().toISOString(),
        kind: "system",
        summary: `Restored snapshot ${summary.label}.`,
        detail: snapshotId
      }
    ];
    const engineSnapshot = join(this.brainDirectory(id), "snapshots", snapshotId, "engine");
    if (await pathExists(join(engineSnapshot, "brain.json"))) {
      const engineMetadata = await readFile(join(engineSnapshot, "brain.json"));
      const engineMetadataValue = JSON.parse(engineMetadata.toString("utf8")) as unknown;
      const hashes = [sha256(engineMetadata)];
      for (const name of ["core.safetensors", "plasticity.safetensors"]) {
        const sourcePath = join(engineSnapshot, name);
        if (await pathExists(sourcePath)) hashes.push(await fileSha256(sourcePath));
      }
      if (await pathExists(join(engineSnapshot, "packed-ternary", "manifest.json"))) {
        const packed = await inspectPackedTernaryDirectory(
          join(engineSnapshot, "packed-ternary"),
          "Snapshot"
        );
        hashes.push(packed.manifestSha256);
      }
      const substrate = await collectSubstrateSnapshot(
        engineSnapshot,
        "substrate/snapshot",
        engineMetadataValue
      );
      if (substrate) hashes.push(sha256(canonicalJson(substrate.pointer)));
      if (summary.engineChecksum && sha256(hashes.join(":")) !== summary.engineChecksum) {
        throw new Error("Neural snapshot checksum validation failed.");
      }
    }
    if (!(await pathExists(join(engineSnapshot, "brain.json")))) {
      return this.save(restored);
    }

    const brainDirectory = this.brainDirectory(id);
    const targetEngine = join(brainDirectory, "engine");
    const stagedEngine = join(brainDirectory, `.engine-${randomUUID()}.restore`);
    const previousEngine = join(brainDirectory, `.engine-${randomUUID()}.previous`);
    const failedEngine = join(brainDirectory, `.engine-${randomUUID()}.failed`);
    let previousMoved = false;
    let promoted = false;
    try {
      if (await pathExists(targetEngine)) {
        // Preserve append-only events, immutable origin, artifacts, and other
        // non-generation state while replacing the neural generation below.
        await cp(targetEngine, stagedEngine, {
          recursive: true,
          force: false,
          errorOnExist: true,
          preserveTimestamps: true
        });
      } else {
        await mkdir(stagedEngine, { recursive: true });
      }
      const metadata = await readFile(join(engineSnapshot, "brain.json"));
      const metadataValue = JSON.parse(metadata.toString("utf8")) as unknown;
      for (const name of ["core.safetensors", "plasticity.safetensors"]) {
        const sourcePath = join(engineSnapshot, name);
        if (!(await pathExists(sourcePath))) continue;
        const hash = await this.storeFileAsBlob(sourcePath);
        await this.linkBlobTo(hash, join(stagedEngine, name));
      }
      await this.copyPackedTernaryDirectory(
        join(engineSnapshot, "packed-ternary"),
        join(stagedEngine, "packed-ternary")
      );
      await this.copySubstrateSnapshot(
        engineSnapshot,
        stagedEngine,
        metadataValue
      );
      // Metadata is the staged generation's final commit record.
      await atomicWrite(join(stagedEngine, "brain.json"), metadata);

      if (await pathExists(targetEngine)) {
        await rename(targetEngine, previousEngine);
        previousMoved = true;
      }
      await rename(stagedEngine, targetEngine);
      promoted = true;
      try {
        const saved = await this.save(restored);
        await rm(previousEngine, { recursive: true, force: true });
        previousMoved = false;
        return saved;
      } catch (error) {
        await rename(targetEngine, failedEngine);
        promoted = false;
        if (previousMoved) {
          await rename(previousEngine, targetEngine);
          previousMoved = false;
        }
        await rm(failedEngine, { recursive: true, force: true });
        throw error;
      }
    } catch (error) {
      if (promoted && (await pathExists(targetEngine))) {
        await rename(targetEngine, failedEngine).catch(() => undefined);
        promoted = false;
      }
      if (previousMoved && (await pathExists(previousEngine))) {
        await rename(previousEngine, targetEngine).catch(() => undefined);
        previousMoved = false;
      }
      throw error;
    } finally {
      await Promise.all([
        rm(stagedEngine, { recursive: true, force: true }),
        rm(failedEngine, { recursive: true, force: true }),
        previousMoved
          ? Promise.resolve()
          : rm(previousEngine, { recursive: true, force: true })
      ]);
    }
  }

  async exportBundle(
    id: string,
    destination: string,
    mode: BrainExportMode = "current"
  ): Promise<void> {
    const currentBrain = await this.get(id);
    const directory = this.brainDirectory(id);
    const redactions: RedactionCounter = { replacements: 0 };
    let portableBrain =
      mode === "origin"
        ? normalizeBrain(JSON.parse(await readFile(join(directory, "origin.json"), "utf8")))
        : clone(currentBrain);
    portableBrain.toolPermissions = (portableBrain.toolPermissions ?? []).map((permission) => ({
      ...permission,
      level: permission.level === "off" ? "off" : "ask"
    }));
    if (mode !== "private-archive") {
      portableBrain.ideas = portableBrain.ideas.map((idea) =>
        idea.source === "document" || idea.source === "import"
          ? { ...idea, statement: undefined }
          : idea
      );
      portableBrain.trainingSources = portableBrain.trainingSources.map((source) => {
        const sanitized = { ...source };
        delete sanitized.path;
        delete sanitized.rawText;
        delete sanitized.blobHash;
        sanitized.rawTextRetained = false;
        return sanitized;
      });
    }
    portableBrain = redactPortableValue(portableBrain, redactions) as BrainDocument;
    const engineDirectory =
      mode === "origin" ? join(directory, "engine", "origin") : join(directory, "engine");
    const engineStatePath = join(engineDirectory, "brain.json");
    const engineMaterialized = await pathExists(engineStatePath);
    const selectedPackedPath = join(engineDirectory, "packed-ternary");
    const selectedPacked =
      engineMaterialized && (await pathExists(join(selectedPackedPath, "manifest.json")))
        ? await inspectPackedTernaryDirectory(selectedPackedPath, "Current")
        : undefined;
    const engineState = engineMaterialized
      ? portableEngineState(
          await readFile(engineStatePath),
          mode === "private-archive",
          redactions
        )
      : strToU8(
          JSON.stringify(
            {
              format: "omni-engine-unmaterialized",
              brain_id: portableBrain.id,
              name: portableBrain.name
            },
            null,
            2
          )
        );
    if (engineMaterialized && !selectedPacked) {
      throw new Error(
        "Materialized OmniCortex state is missing its verified packed ternary inference shards."
      );
    }
    const corePath = join(engineDirectory, "core.safetensors");
    const plasticityPath = join(engineDirectory, "plasticity.safetensors");
    const coreExists = await pathExists(corePath);
    const plasticityExists = await pathExists(plasticityPath);
    const coreFallback = coreExists
      ? undefined
      : validEmptySafetensors("Neural core has not been materialized by the Python worker.");
    const plasticityFallback = plasticityExists
      ? undefined
      : validEmptySafetensors("Plastic state is represented by state/brain.json.");
    if (coreExists) await assertSafeTensorsFile(corePath, "core.safetensors");
    else assertSafeTensors(coreFallback!, "core.safetensors");
    if (plasticityExists) {
      await assertSafeTensorsFile(plasticityPath, "plastic.safetensors");
    } else assertSafeTensors(plasticityFallback!, "plastic.safetensors");
    const entries: Record<string, StreamingZipSource> = {
      "model-card.md": {
        name: "model-card.md",
        contents: strToU8(
        `# ${portableBrain.name}\n\nOmniCortex brain ${portableBrain.id}.\n\n` +
          `Preset: ${portableBrain.config.preset}\n\n` +
          `Memory recipe: ${portableBrain.config.memoryRecipe ?? "human-consolidation"}\n`
        )
      },
      "state/brain.json": {
        name: "state/brain.json",
        contents: strToU8(JSON.stringify(portableBrain, null, 2))
      },
      "state/engine.json": {
        name: "state/engine.json",
        contents: engineState
      },
      "tensors/core.safetensors": {
        name: "tensors/core.safetensors",
        ...(coreExists ? { sourcePath: corePath } : { contents: coreFallback })
      },
      "tensors/plastic.safetensors": {
        name: "tensors/plastic.safetensors",
        ...(plasticityExists
          ? { sourcePath: plasticityPath }
          : { contents: plasticityFallback })
      }
    };
    if (selectedPacked) {
      for (const [name, sourcePath] of selectedPacked.files) {
        const archivePath = `packed/current/${name}`;
        entries[archivePath] = { name: archivePath, sourcePath };
      }
    }
    let originBrain = normalizeBrain(
      JSON.parse(await readFile(join(directory, "origin.json"), "utf8"))
    );
    originBrain.toolPermissions = (originBrain.toolPermissions ?? []).map((permission) => ({
      ...permission,
      level: permission.level === "off" ? "off" : "ask"
    }));
    originBrain.trainingSources = originBrain.trainingSources.map((source) => {
      const sanitized = { ...source };
      delete sanitized.path;
      delete sanitized.rawText;
      delete sanitized.blobHash;
      sanitized.rawTextRetained = false;
      return sanitized;
    });
    originBrain.ideas = originBrain.ideas.map((idea) =>
      idea.source === "document" || idea.source === "import"
        ? { ...idea, statement: undefined }
        : idea
    );
    originBrain = redactPortableValue(originBrain, redactions) as BrainDocument;
    const immutableEngine = join(directory, "engine", "origin");
    const immutableStatePath = join(immutableEngine, "brain.json");
    const immutableState = (await pathExists(immutableStatePath))
      ? portableEngineState(await readFile(immutableStatePath), false, redactions)
      : mode === "origin"
        ? engineState
        : strToU8(
            JSON.stringify(
              {
                format: "omni-engine-unmaterialized",
                brain_id: originBrain.id,
                name: originBrain.name
              },
              null,
              2
            )
          );
    const immutableCorePath = join(immutableEngine, "core.safetensors");
    const immutablePlasticPath = join(immutableEngine, "plasticity.safetensors");
    const immutableCoreExists = await pathExists(immutableCorePath);
    const immutablePlasticExists = await pathExists(immutablePlasticPath);
    if (immutableCoreExists) {
      await assertSafeTensorsFile(immutableCorePath, "origin core.safetensors");
    }
    if (immutablePlasticExists) {
      await assertSafeTensorsFile(
        immutablePlasticPath,
        "origin plastic.safetensors"
      );
    }
    const immutablePackedPath = join(immutableEngine, "packed-ternary");
    const immutablePacked =
      engineMaterialized &&
      (await pathExists(join(immutablePackedPath, "manifest.json")))
        ? await inspectPackedTernaryDirectory(immutablePackedPath, "Origin")
        : mode === "origin"
          ? selectedPacked
          : undefined;
    if (engineMaterialized && !immutablePacked) {
      throw new Error(
        "Materialized OmniCortex state is missing its immutable-origin packed ternary shards."
      );
    }
    const references =
      mode === "referenced"
        ? {
            currentCore: coreExists
              ? await this.storeFileAsBlob(corePath)
              : await this.storeBlob(Buffer.from(coreFallback!)),
            currentPlasticity: plasticityExists
              ? await this.storeFileAsBlob(plasticityPath)
              : await this.storeBlob(Buffer.from(plasticityFallback!)),
            originCore: immutableCoreExists
              ? await this.storeFileAsBlob(immutableCorePath)
              : coreExists
                ? await this.storeFileAsBlob(corePath)
                : await this.storeBlob(Buffer.from(coreFallback!)),
            originPlasticity: immutablePlasticExists
              ? await this.storeFileAsBlob(immutablePlasticPath)
              : plasticityExists
                ? await this.storeFileAsBlob(plasticityPath)
                : await this.storeBlob(Buffer.from(plasticityFallback!))
          }
        : undefined;
    if (references) {
      entries["tensors/core.safetensors"] = {
        name: "tensors/core.safetensors",
        contents: validEmptySafetensors(
          `Local content reference ${references.currentCore}`
        )
      };
      entries["tensors/plastic.safetensors"] = {
        name: "tensors/plastic.safetensors",
        contents: validEmptySafetensors(
          `Local content reference ${references.currentPlasticity}`
        )
      };
    }
    entries["origin/state/brain.json"] = {
      name: "origin/state/brain.json",
      contents: strToU8(JSON.stringify(originBrain, null, 2))
    };
    entries["origin/state/engine.json"] = {
      name: "origin/state/engine.json",
      contents: immutableState
    };
    entries["origin/tensors/core.safetensors"] = {
      name: "origin/tensors/core.safetensors",
      ...(references
        ? {
            contents: validEmptySafetensors(
              `Local content reference ${references.originCore}`
            )
          }
        : immutableCoreExists
          ? { sourcePath: immutableCorePath }
          : coreExists
            ? { sourcePath: corePath }
            : { contents: coreFallback })
    };
    entries["origin/tensors/plastic.safetensors"] = {
      name: "origin/tensors/plastic.safetensors",
      ...(references
        ? {
            contents: validEmptySafetensors(
              `Local content reference ${references.originPlasticity}`
            )
          }
        : immutablePlasticExists
          ? { sourcePath: immutablePlasticPath }
          : plasticityExists
            ? { sourcePath: plasticityPath }
            : { contents: plasticityFallback })
    };
    if (immutablePacked) {
      for (const [name, sourcePath] of immutablePacked.files) {
        const archivePath = `packed/origin/${name}`;
        entries[archivePath] = { name: archivePath, sourcePath };
      }
    }
    let packedReferences:
      | {
          current: Record<string, string>;
          origin: Record<string, string>;
        }
      | undefined;
    if (references && selectedPacked && immutablePacked) {
      packedReferences = { current: {}, origin: {} };
      for (const [scope, packed] of [
        ["current", selectedPacked],
        ["origin", immutablePacked]
      ] as const) {
        for (const [name, sourcePath] of packed.files) {
          const hash = await this.storeFileAsBlob(sourcePath);
          packedReferences[scope][name] = hash;
          const archivePath = `packed/${scope}/${name}`;
          entries[archivePath] = {
            name: archivePath,
            contents: strToU8(`Local content reference ${hash}\n`)
          };
        }
      }
    }
    for (const source of mode === "private-archive" ? portableBrain.trainingSources : []) {
      if (!source.blobHash || entries[`blobs/${source.blobHash}`]) continue;
      if (!/^[a-f0-9]{64}$/.test(source.blobHash)) {
        throw new Error("Private archive contains an invalid content-addressed blob.");
      }
      const blobPath = join(this.root, ".blobs", source.blobHash);
      if ((await streamFileSha256(blobPath)) !== source.blobHash) {
        throw new Error("Content-addressed blob checksum failed.");
      }
      await assertFileContainsNoPortableSecrets(blobPath, source.name);
      const archivePath = `blobs/${source.blobHash}`;
      entries[archivePath] = { name: archivePath, sourcePath: blobPath };
    }
    const currentEngineMetadata = engineMaterialized
      ? (JSON.parse(Buffer.from(engineState).toString("utf8")) as unknown)
      : undefined;
    const originEngineMetadata =
      Buffer.from(immutableState).toString("utf8").includes("omni-cortex-engine")
        ? (JSON.parse(Buffer.from(immutableState).toString("utf8")) as unknown)
        : undefined;
    const [currentSubstrate, originSubstrate] = await Promise.all([
      collectSubstrateSnapshot(
        engineDirectory,
        "substrate/current",
        currentEngineMetadata
      ),
      collectSubstrateSnapshot(
        immutableEngine,
        "substrate/origin",
        originEngineMetadata
      )
    ]);
    for (const snapshot of [currentSubstrate, originSubstrate]) {
      for (const source of snapshot?.sources ?? []) entries[source.name] = source;
    }
    const fileRecords: Record<string, { sha256: string; bytes: number }> = {};
    for (const [path, source] of Object.entries(entries)) {
      if (source.contents) {
        fileRecords[path] = {
          sha256: sha256(Buffer.from(source.contents)),
          bytes: source.contents.byteLength
        };
      } else if (source.sourcePath) {
        const info = await lstat(source.sourcePath);
        if (!info.isFile() || info.isSymbolicLink()) {
          throw new Error(`Bundle source ${path} is not a regular file.`);
        }
        fileRecords[path] = {
          sha256: await streamFileSha256(source.sourcePath),
          bytes: info.size
        };
      } else throw new Error(`Bundle source ${path} is empty.`);
    }
    const manifest: OmniManifest = {
      format: BUNDLE_FORMAT,
      formatVersion: BUNDLE_VERSION,
      releaseFormat: STABLE_RELEASE_FORMAT,
      architecture: "OmniCortex",
      architectureSchemaVersion: portableBrain.schemaVersion,
      exportedAt: new Date().toISOString(),
      brain: {
        id: portableBrain.id,
        name: portableBrain.name,
        lineage: portableBrain.lineage
      },
      mode:
        mode === "origin"
          ? "origin-portable"
          : mode === "private-archive"
            ? "private-archive"
            : mode === "referenced"
              ? "referenced-local"
            : "current-portable",
      engineMaterialized,
      memoryRecipe: portableBrain.config.memoryRecipe ?? "human-consolidation",
      rawEpisodesPresent: portableBrain.trainingSources.some((source) => source.rawTextRetained),
      quantization: "ternary-effective",
      packedTernary:
        selectedPacked && immutablePacked
          ? {
              format: "omni-packed-ternary",
              formatVersion: 1,
              currentManifestSha256: selectedPacked.manifestSha256,
              originManifestSha256: immutablePacked.manifestSha256,
              currentTensorCount: selectedPacked.tensorCount,
              originTensorCount: immutablePacked.tensorCount,
              references: packedReferences
            }
          : undefined,
      secretRedaction: {
        version: 1,
        replacements: redactions.replacements
      },
      licenseLedger: {
        application: "PolyForm-Noncommercial-1.0.0-or-commercial-license",
        sources: portableBrain.trainingSources.map((source) => ({
          name: source.name,
          provenanceUrl: source.provenanceUrl,
          license: source.license ?? "Undeclared; verify before redistribution",
          licenseUrl: source.licenseUrl
        }))
      },
      references,
      files: fileRecords
    };
    const manifestContents = strToU8(JSON.stringify(manifest, null, 2));
    entries["manifest.json"] = {
      name: "manifest.json",
      contents: manifestContents
    };
    const checksumRecords = {
      ...fileRecords,
      "manifest.json": {
        sha256: sha256(Buffer.from(manifestContents)),
        bytes: manifestContents.byteLength
      }
    };
    entries["checksums.sha256"] = {
      name: "checksums.sha256",
      contents: strToU8(
        Object.entries(checksumRecords)
        .map(([path, descriptor]) => `${descriptor.sha256}  ${path}`)
        .sort()
        .join("\n") + "\n"
      )
    };
    await writeStreamingZip(
      destination,
      Object.values(entries).sort((left, right) => left.name.localeCompare(right.name))
    );
  }

  async importBundle(path: string): Promise<BrainDocument> {
    await this.initialize();
    const info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink()) {
      throw new Error("The selected .omni bundle is not a regular file.");
    }
    const extracted = await extractStreamingZip(
      path,
      join(this.root, ".imports")
    );
    try {
      return await this.importExtractedBundle(extracted, basename(path));
    } finally {
      await rm(extracted.root, { recursive: true, force: true });
    }
  }

  async importBundleBuffer(contents: Buffer, sourceLabel = "download.omni"): Promise<BrainDocument> {
    await this.initialize();
    const temporary = await mkdtemp(join(this.root, ".omni-buffer-"));
    const bundlePath = join(temporary, "buffer.omni");
    try {
      await writeFile(bundlePath, contents, { flag: "wx", mode: 0o600 });
      const extracted = await extractStreamingZip(
        bundlePath,
        join(this.root, ".imports")
      );
      try {
        return await this.importExtractedBundle(extracted, sourceLabel);
      } finally {
        await rm(extracted.root, { recursive: true, force: true });
      }
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
  }

  private async importExtractedBundle(
    archive: ExtractedZipArchive,
    sourceLabel: string
  ): Promise<BrainDocument> {
    const names = new Set(archive.entries.keys());
    const entryPath = (name: string): string => {
      const entry = archive.entries.get(name);
      if (!entry) throw new Error(`The .omni bundle is missing ${name}.`);
      return entry.path;
    };
    for (const required of [
      "manifest.json",
      "model-card.md",
      "checksums.sha256",
      "state/brain.json",
      "state/engine.json",
      "tensors/core.safetensors",
      "tensors/plastic.safetensors",
      "origin/state/brain.json",
      "origin/state/engine.json",
      "origin/tensors/core.safetensors",
      "origin/tensors/plastic.safetensors"
    ]) {
      if (!names.has(required)) {
        throw new Error(`The .omni bundle is missing ${required}.`);
      }
    }
    const checksums = parseChecksumFile(
      await readFile(entryPath("checksums.sha256"), "utf8")
    );
    for (const [path, expected] of checksums) {
      const entry = archive.entries.get(path);
      if (!entry) throw new Error(`Checksum references missing file ${path}.`);
      if ((await streamFileSha256(entry.path)) !== expected) {
        throw new Error(`Checksum validation failed for ${path}.`);
      }
    }
    for (const path of names) {
      assertAllowedBundlePath(path);
      if (path !== "checksums.sha256" && !checksums.has(path)) {
        throw new Error(`The .omni bundle has no checksum for ${path}.`);
      }
    }
    let manifestValue: unknown;
    try {
      manifestValue = JSON.parse(await readFile(entryPath("manifest.json"), "utf8"));
    } catch {
      throw new Error("manifest.json is invalid.");
    }
    if (
      !isRecord(manifestValue) ||
      manifestValue.format !== BUNDLE_FORMAT ||
      manifestValue.formatVersion !== BUNDLE_VERSION ||
      manifestValue.releaseFormat !== STABLE_RELEASE_FORMAT
    ) {
      throw new Error(
        "Unsupported or beta .omni bundle; stable Omni AGI Studio v1 format is required."
      );
    }
    if (
      manifestValue.architecture !== "OmniCortex" ||
      manifestValue.architectureSchemaVersion !== BRAIN_SCHEMA_VERSION ||
      manifestValue.quantization !== "ternary-effective" ||
      !["current-portable", "origin-portable", "private-archive", "referenced-local"].includes(
        String(manifestValue.mode)
      )
    ) {
      throw new Error("The .omni bundle targets an incompatible architecture schema.");
    }
    const declaredEngineMaterialized = manifestValue.engineMaterialized === true;
    let packedDeclaration: OmniManifest["packedTernary"];
    if (manifestValue.packedTernary !== undefined) {
      const packed = manifestValue.packedTernary;
      if (
        !isRecord(packed) ||
        packed.format !== "omni-packed-ternary" ||
        packed.formatVersion !== 1 ||
        typeof packed.currentManifestSha256 !== "string" ||
        !/^[a-f0-9]{64}$/.test(packed.currentManifestSha256) ||
        typeof packed.originManifestSha256 !== "string" ||
        !/^[a-f0-9]{64}$/.test(packed.originManifestSha256) ||
        typeof packed.currentTensorCount !== "number" ||
        !Number.isSafeInteger(packed.currentTensorCount) ||
        packed.currentTensorCount < 1 ||
        typeof packed.originTensorCount !== "number" ||
        !Number.isSafeInteger(packed.originTensorCount) ||
        packed.originTensorCount < 1
      ) {
        throw new Error("The .omni bundle has an invalid packed ternary declaration.");
      }
      let packedReferenceDeclaration:
        | {
            current: Record<string, string>;
            origin: Record<string, string>;
          }
        | undefined;
      if (packed.references !== undefined) {
        if (
          !isRecord(packed.references) ||
          !isRecord(packed.references.current) ||
          !isRecord(packed.references.origin)
        ) {
          throw new Error("The packed ternary local references are invalid.");
        }
        const normalizeReferences = (
          value: Record<string, unknown>
        ): Record<string, string> => {
          const result: Record<string, string> = {};
          for (const [name, hash] of Object.entries(value)) {
            if (
              !["manifest.json", "manifest.sha256"].includes(name) &&
              !/^ternary-[0-9]{5,}-[a-f0-9]{16}\.bin$/.test(name)
            ) {
              throw new Error("A packed ternary local reference has an unsafe name.");
            }
            if (typeof hash !== "string" || !/^[a-f0-9]{64}$/.test(hash)) {
              throw new Error("A packed ternary local reference has an invalid hash.");
            }
            result[name] = hash;
          }
          return result;
        };
        packedReferenceDeclaration = {
          current: normalizeReferences(packed.references.current),
          origin: normalizeReferences(packed.references.origin)
        };
      }
      packedDeclaration = {
        format: "omni-packed-ternary",
        formatVersion: 1,
        currentManifestSha256: packed.currentManifestSha256,
        originManifestSha256: packed.originManifestSha256,
        currentTensorCount: packed.currentTensorCount,
        originTensorCount: packed.originTensorCount,
        references: packedReferenceDeclaration
      };
    }
    if (declaredEngineMaterialized && !packedDeclaration) {
      throw new Error(
        "A materialized stable v1 brain must contain packed ternary inference shards."
      );
    }
    if (
      packedDeclaration?.references &&
      manifestValue.mode !== "referenced-local"
    ) {
      throw new Error("Portable bundles may not contain packed ternary local references.");
    }
    if (
      manifestValue.mode === "referenced-local" &&
      declaredEngineMaterialized &&
      !packedDeclaration?.references
    ) {
      throw new Error("The local referenced bundle has no packed ternary references.");
    }
    if (
      !isRecord(manifestValue.secretRedaction) ||
      manifestValue.secretRedaction.version !== 1 ||
      typeof manifestValue.secretRedaction.replacements !== "number"
    ) {
      throw new Error("The .omni bundle does not declare a supported secret-redaction policy.");
    }
    if (
      !isRecord(manifestValue.licenseLedger) ||
      typeof manifestValue.licenseLedger.application !== "string" ||
      !Array.isArray(manifestValue.licenseLedger.sources) ||
      manifestValue.licenseLedger.sources.some(
        (source) =>
          !isRecord(source) ||
          typeof source.name !== "string" ||
          typeof source.license !== "string" ||
          source.name.length > 4_000 ||
          source.license.length > 4_000 ||
          (source.provenanceUrl !== undefined && typeof source.provenanceUrl !== "string") ||
          (source.licenseUrl !== undefined && typeof source.licenseUrl !== "string")
      )
    ) {
      throw new Error("The .omni bundle does not contain a valid license ledger.");
    }
    const manifestFiles = isRecord(manifestValue.files) ? manifestValue.files : {};
    for (const [path, descriptor] of Object.entries(manifestFiles)) {
      if (
        !isRecord(descriptor) ||
        typeof descriptor.sha256 !== "string" ||
        !/^[a-f0-9]{64}$/.test(descriptor.sha256) ||
        typeof descriptor.bytes !== "number" ||
        !Number.isSafeInteger(descriptor.bytes) ||
        descriptor.bytes < 0
      ) {
        throw new Error(`Manifest descriptor for ${path} is invalid.`);
      }
      assertAllowedBundlePath(path);
      const entry = archive.entries.get(path);
      if (
        !entry ||
        entry.uncompressedBytes !== descriptor.bytes ||
        checksums.get(path) !== descriptor.sha256
      ) {
        throw new Error(`Manifest checksum validation failed for ${path}.`);
      }
    }
    for (const path of names) {
      if (path === "manifest.json" || path === "checksums.sha256") continue;
      if (!Object.hasOwn(manifestFiles, path)) {
        throw new Error(`Manifest is missing a descriptor for ${path}.`);
      }
    }
    const packedOverrides = {
      current: new Map<string, string>(),
      origin: new Map<string, string>()
    };
    const tensorPaths: Record<string, string> = {
      "tensors/core.safetensors": entryPath("tensors/core.safetensors"),
      "tensors/plastic.safetensors": entryPath("tensors/plastic.safetensors"),
      "origin/tensors/core.safetensors": entryPath(
        "origin/tensors/core.safetensors"
      ),
      "origin/tensors/plastic.safetensors": entryPath(
        "origin/tensors/plastic.safetensors"
      )
    };
    let resolvedReferences: OmniManifest["references"];
    if (manifestValue.mode === "referenced-local") {
      if (!isRecord(manifestValue.references)) {
        throw new Error("The local referenced bundle has no tensor references.");
      }
      const mappings = [
        ["currentCore", "tensors/core.safetensors"],
        ["currentPlasticity", "tensors/plastic.safetensors"],
        ["originCore", "origin/tensors/core.safetensors"],
        ["originPlasticity", "origin/tensors/plastic.safetensors"]
      ] as const;
      for (const [key, path] of mappings) {
        const hash = manifestValue.references[key];
        if (typeof hash !== "string" || !/^[a-f0-9]{64}$/.test(hash)) {
          throw new Error("The local referenced bundle contains an invalid tensor reference.");
        }
        const blobPath = join(this.root, ".blobs", hash);
        try {
          if ((await streamFileSha256(blobPath)) !== hash) throw new Error("checksum");
          tensorPaths[path] = blobPath;
        } catch {
          throw new Error(
            `Local tensor reference ${hash.slice(0, 12)}… is unavailable on this installation.`
          );
        }
      }
      resolvedReferences = {
        currentCore: manifestValue.references.currentCore as string,
        currentPlasticity: manifestValue.references.currentPlasticity as string,
        originCore: manifestValue.references.originCore as string,
        originPlasticity: manifestValue.references.originPlasticity as string
      };
    } else if (manifestValue.references !== undefined) {
      throw new Error("Portable bundles may not contain local tensor references.");
    }
    if (packedDeclaration?.references) {
      for (const scope of ["current", "origin"] as const) {
        const referencesForScope: Record<string, string> =
          packedDeclaration.references[scope];
        for (const [name, hash] of Object.entries(referencesForScope) as Array<
          [string, string]
        >) {
          const path = `packed/${scope}/${name}`;
          if (!archive.entries.has(path)) {
            throw new Error(`The local referenced bundle is missing ${path}.`);
          }
          try {
            const blobPath = join(this.root, ".blobs", hash);
            if ((await streamFileSha256(blobPath)) !== hash) throw new Error("checksum");
            packedOverrides[scope].set(name, blobPath);
          } catch {
            throw new Error(
              `Local packed ternary reference ${hash.slice(0, 12)}… is unavailable on this installation.`
            );
          }
        }
      }
    }
    let verifiedPackedCurrent: StreamingPackedTernaryDirectory | undefined;
    let verifiedPackedOrigin: StreamingPackedTernaryDirectory | undefined;
    const bundledCurrentNames = [...names].filter((name) =>
      name.startsWith("packed/current/")
    );
    const bundledOriginNames = [...names].filter((name) =>
      name.startsWith("packed/origin/")
    );
    if (packedDeclaration) {
      verifiedPackedCurrent = await inspectPackedTernaryDirectory(
        join(archive.root, "packed", "current"),
        "Current bundle",
        packedOverrides.current
      );
      verifiedPackedOrigin = await inspectPackedTernaryDirectory(
        join(archive.root, "packed", "origin"),
        "Origin bundle",
        packedOverrides.origin
      );
      if (
        verifiedPackedCurrent.manifestSha256 !==
          packedDeclaration.currentManifestSha256 ||
        verifiedPackedOrigin.manifestSha256 !==
          packedDeclaration.originManifestSha256 ||
        verifiedPackedCurrent.tensorCount !== packedDeclaration.currentTensorCount ||
        verifiedPackedOrigin.tensorCount !== packedDeclaration.originTensorCount
      ) {
        throw new Error("Packed ternary bundle metadata does not match its manifest.");
      }
    } else if (
      bundledCurrentNames.length > 0 ||
      bundledOriginNames.length > 0
    ) {
      throw new Error("The .omni bundle contains undeclared packed ternary data.");
    }
    await Promise.all([
      assertSafeTensorsFile(tensorPaths["tensors/core.safetensors"]!, "core.safetensors"),
      assertSafeTensorsFile(
        tensorPaths["tensors/plastic.safetensors"]!,
        "plastic.safetensors"
      ),
      assertSafeTensorsFile(
        tensorPaths["origin/tensors/core.safetensors"]!,
        "origin core.safetensors"
      ),
      assertSafeTensorsFile(
        tensorPaths["origin/tensors/plastic.safetensors"]!,
        "origin plastic.safetensors"
      )
    ]);
    for (const path of names) {
      if (!path.startsWith("blobs/")) continue;
      const hash = path.slice("blobs/".length);
      if (
        !/^[a-f0-9]{64}$/.test(hash) ||
        (await streamFileSha256(entryPath(path))) !== hash
      ) {
        throw new Error(`Content-addressed blob validation failed for ${path}.`);
      }
    }
    let brainValue: unknown;
    let originBrainValue: unknown;
    let engineValue: unknown;
    let originEngineValue: unknown;
    try {
      brainValue = JSON.parse(await readFile(entryPath("state/brain.json"), "utf8"));
      originBrainValue = JSON.parse(
        await readFile(entryPath("origin/state/brain.json"), "utf8")
      );
      engineValue = JSON.parse(await readFile(entryPath("state/engine.json"), "utf8"));
      originEngineValue = JSON.parse(
        await readFile(entryPath("origin/state/engine.json"), "utf8")
      );
    } catch {
      throw new Error("A required brain or engine state document is invalid.");
    }

    const imported = normalizeBrain(brainValue);
    const importedOrigin = normalizeBrain(originBrainValue);
    const engineMaterialized = manifestValue.engineMaterialized === true;
    if (
      engineMaterialized &&
      (!isRecord(engineValue) ||
        engineValue.format !== "omni-cortex-engine" ||
        engineValue.schema_version !== 1 ||
        engineValue.release_format !== STABLE_RELEASE_FORMAT ||
        !isRecord(originEngineValue) ||
        originEngineValue.format !== "omni-cortex-engine" ||
        originEngineValue.schema_version !== 1 ||
        originEngineValue.release_format !== STABLE_RELEASE_FORMAT)
    ) {
      throw new Error("Materialized engine state is invalid or belongs to the beta format.");
    }
    const [currentSubstratePaths, originSubstratePaths] = await Promise.all([
      validateExtractedSubstrateSnapshot(
        archive,
        "substrate/current",
        engineValue
      ),
      validateExtractedSubstrateSnapshot(
        archive,
        "substrate/origin",
        originEngineValue
      )
    ]);
    if (isRecord(engineValue)) engineValue.brain_id = imported.id;
    if (isRecord(originEngineValue)) originEngineValue.brain_id = imported.id;
    if (await pathExists(this.brainDirectory(imported.id))) {
      const previousId = imported.id;
      imported.id = randomUUID();
      imported.lineage = {
        parentId: previousId,
        rootId: imported.lineage.rootId,
        generation: imported.lineage.generation + 1
      };
    }
    if (isRecord(engineValue)) {
      engineValue.brain_id = imported.id;
      engineValue.name = imported.name;
    }
    if (isRecord(originEngineValue)) {
      originEngineValue.brain_id = imported.id;
      originEngineValue.name = imported.name;
    }
    importedOrigin.id = imported.id;
    importedOrigin.name = imported.name;
    importedOrigin.config.name = imported.name;
    importedOrigin.lineage = imported.lineage;
    importedOrigin.originChecksum = undefined;
    const importedOriginChecksum = sha256(JSON.stringify(importedOrigin));
    importedOrigin.originChecksum = importedOriginChecksum;
    imported.originChecksum = importedOriginChecksum;
    imported.name = imported.name.slice(0, 120);
    imported.config.name = imported.name;
    imported.createdAt = new Date().toISOString();
    imported.updatedAt = imported.createdAt;
    imported.journal = [
      ...(imported.journal ?? []),
      {
        id: randomUUID(),
        createdAt: imported.createdAt,
        kind: "system",
        summary: `Imported from ${basename(sourceLabel)}.`
      }
    ];
    const directory = this.brainDirectory(imported.id);
    const materializeFile = async (source: string, destination: string): Promise<void> => {
      const hash = await this.storeFileAsBlob(source);
      await this.linkBlobTo(hash, destination);
    };
    const installSubstrate = async (
      prefix: string,
      paths: Set<string>,
      destination: string
    ): Promise<void> => {
      for (const relative of [...paths].sort()) {
        await materializeFile(
          entryPath(`${prefix}/${relative}`),
          join(destination, ...relative.split("/"))
        );
      }
    };
    try {
      await Promise.all([
        mkdir(join(directory, "snapshots"), { recursive: true }),
        mkdir(join(directory, "engine"), { recursive: true })
      ]);
      await atomicWrite(this.documentPath(imported.id), JSON.stringify(imported, null, 2));
      if (engineMaterialized) {
        if (resolvedReferences) {
          await Promise.all([
            this.linkBlobTo(
              resolvedReferences.currentCore,
              join(directory, "engine", "core.safetensors")
            ),
            this.linkBlobTo(
              resolvedReferences.currentPlasticity,
              join(directory, "engine", "plasticity.safetensors")
            )
          ]);
        } else {
          await Promise.all([
            materializeFile(
              tensorPaths["tensors/core.safetensors"]!,
              join(directory, "engine", "core.safetensors")
            ),
            materializeFile(
              tensorPaths["tensors/plastic.safetensors"]!,
              join(directory, "engine", "plasticity.safetensors")
            )
          ]);
        }
      }
      if (engineMaterialized && verifiedPackedCurrent) {
        const packedDirectory = join(directory, "engine", "packed-ternary");
        await mkdir(packedDirectory, { recursive: true });
        for (const [name, sourcePath] of verifiedPackedCurrent.files) {
          const referenceHash = packedDeclaration?.references?.current[name];
          if (referenceHash) {
            await this.linkBlobTo(referenceHash, join(packedDirectory, name));
          } else {
            await materializeFile(sourcePath, join(packedDirectory, name));
          }
        }
      }
      if (currentSubstratePaths.size > 0) {
        await installSubstrate(
          "substrate/current",
          currentSubstratePaths,
          join(directory, "engine", "substrate")
        );
      }
      if (engineMaterialized) {
        await atomicWrite(
          join(directory, "engine", "brain.json"),
          JSON.stringify(engineValue, null, 2)
        );
      }
      if (isRecord(originEngineValue) && originEngineValue.format === "omni-cortex-engine") {
        if (resolvedReferences) {
          await Promise.all([
            this.linkBlobTo(
              resolvedReferences.originCore,
              join(directory, "engine", "origin", "core.safetensors")
            ),
            this.linkBlobTo(
              resolvedReferences.originPlasticity,
              join(directory, "engine", "origin", "plasticity.safetensors")
            )
          ]);
        } else {
          await Promise.all([
            materializeFile(
              tensorPaths["origin/tensors/core.safetensors"]!,
              join(directory, "engine", "origin", "core.safetensors")
            ),
            materializeFile(
              tensorPaths["origin/tensors/plastic.safetensors"]!,
              join(directory, "engine", "origin", "plasticity.safetensors")
            )
          ]);
        }
      }
      if (
        isRecord(originEngineValue) &&
        originEngineValue.format === "omni-cortex-engine" &&
        verifiedPackedOrigin
      ) {
        const packedDirectory = join(
          directory,
          "engine",
          "origin",
          "packed-ternary"
        );
        await mkdir(packedDirectory, { recursive: true });
        for (const [name, sourcePath] of verifiedPackedOrigin.files) {
          const referenceHash = packedDeclaration?.references?.origin[name];
          if (referenceHash) {
            await this.linkBlobTo(referenceHash, join(packedDirectory, name));
          } else {
            await materializeFile(sourcePath, join(packedDirectory, name));
          }
        }
      }
      if (originSubstratePaths.size > 0) {
        await installSubstrate(
          "substrate/origin",
          originSubstratePaths,
          join(directory, "engine", "origin", "substrate")
        );
      }
      if (isRecord(originEngineValue) && originEngineValue.format === "omni-cortex-engine") {
        await atomicWrite(
          join(directory, "engine", "origin", "brain.json"),
          JSON.stringify(originEngineValue, null, 2)
        );
      }
      for (const path of names) {
        if (!path.startsWith("blobs/")) continue;
        const expected = path.slice("blobs/".length);
        const stored = await this.storeFileAsBlob(entryPath(path));
        if (stored !== expected) {
          throw new Error(`Content-addressed blob validation failed for ${path}.`);
        }
      }
      await writeFile(join(directory, "origin.json"), JSON.stringify(importedOrigin, null, 2), {
        encoding: "utf8",
        flag: "wx",
        mode: 0o600
      });
      return clone(imported);
    } catch (error) {
      await rm(directory, { recursive: true, force: true });
      throw error;
    }
  }
}
