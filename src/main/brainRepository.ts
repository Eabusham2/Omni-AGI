import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import {
  access,
  copyFile,
  link,
  mkdir,
  open,
  readFile,
  readdir,
  realpath,
  rename,
  rm,
  stat,
  writeFile
} from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { strFromU8, strToU8, unzipSync, zipSync } from "fflate";
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

const MAX_BUNDLE_BYTES = 512 * 1024 * 1024;
const MAX_UNCOMPRESSED_BUNDLE_BYTES = 1024 * 1024 * 1024;
const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const BUNDLE_FORMAT = "omni-brain";
const BUNDLE_VERSION = 1;
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

interface PackedTernaryDirectory {
  manifestSha256: string;
  tensorCount: number;
  files: Record<string, Uint8Array>;
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

interface ZipDirectoryEntry {
  name: string;
  uncompressedBytes: number;
}

function inspectZipDirectory(contents: Buffer): ZipDirectoryEntry[] {
  const minimumEocd = 22;
  if (contents.byteLength < minimumEocd) throw new Error("The .omni ZIP container is truncated.");
  let eocd = -1;
  const searchStart = Math.max(0, contents.byteLength - 65_557);
  for (let index = contents.byteLength - minimumEocd; index >= searchStart; index -= 1) {
    if (contents.readUInt32LE(index) === 0x06054b50) {
      eocd = index;
      break;
    }
  }
  if (eocd < 0) throw new Error("The .omni file is not a supported ZIP container.");
  const entries = contents.readUInt16LE(eocd + 10);
  if (entries > 4_096) throw new Error("The .omni bundle contains too many files.");
  const centralSize = contents.readUInt32LE(eocd + 12);
  const centralOffset = contents.readUInt32LE(eocd + 16);
  if (centralOffset + centralSize > eocd) throw new Error("The .omni ZIP directory is invalid.");
  const seen = new Set<string>();
  const result: ZipDirectoryEntry[] = [];
  let total = 0;
  let cursor = centralOffset;
  for (let index = 0; index < entries; index += 1) {
    if (cursor + 46 > contents.byteLength || contents.readUInt32LE(cursor) !== 0x02014b50) {
      throw new Error("The .omni ZIP central directory is invalid.");
    }
    const flags = contents.readUInt16LE(cursor + 8);
    const compression = contents.readUInt16LE(cursor + 10);
    const uncompressedBytes = contents.readUInt32LE(cursor + 24);
    const nameLength = contents.readUInt16LE(cursor + 28);
    const extraLength = contents.readUInt16LE(cursor + 30);
    const commentLength = contents.readUInt16LE(cursor + 32);
    const externalAttributes = contents.readUInt32LE(cursor + 38);
    const nameStart = cursor + 46;
    const nameEnd = nameStart + nameLength;
    if (nameEnd > contents.byteLength) throw new Error("The .omni ZIP filename is truncated.");
    const name = contents.subarray(nameStart, nameEnd).toString("utf8");
    if ((flags & 0x1) !== 0) throw new Error("Encrypted .omni entries are not supported.");
    if (compression !== 0 && compression !== 8) {
      throw new Error(`Unsupported ZIP compression for ${name}.`);
    }
    const unixMode = externalAttributes >>> 16;
    if ((unixMode & 0o170000) === 0o120000) {
      throw new Error(`Symbolic links are not allowed in .omni bundles: ${name}`);
    }
    assertAllowedBundlePath(name);
    if (seen.has(name)) throw new Error(`Duplicate path in .omni bundle: ${name}`);
    seen.add(name);
    total += uncompressedBytes;
    if (uncompressedBytes > MAX_BUNDLE_BYTES || total > MAX_UNCOMPRESSED_BUNDLE_BYTES) {
      throw new Error("The .omni bundle expands beyond its safe size limit.");
    }
    result.push({ name, uncompressedBytes });
    cursor = nameEnd + extraLength + commentLength;
  }
  return result;
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
  merged.workingMemorySlots = Math.round(
    boundNumber(merged.workingMemorySlots, DEFAULT_CONFIG.workingMemorySlots, 1, 4_096)
  );
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

function verifyPackedTernaryFiles(
  files: Record<string, Uint8Array>,
  label: string
): PackedTernaryDirectory {
  const manifestBytes = Buffer.from(files["manifest.json"] ?? []);
  const checksumBytes = Buffer.from(files["manifest.sha256"] ?? []);
  const expectedManifestHash = checksumBytes.toString("ascii").trim();
  if (
    !/^[a-f0-9]{64}$/.test(expectedManifestHash) ||
    sha256(manifestBytes) !== expectedManifestHash
  ) {
    throw new Error(`${label} packed ternary manifest checksum failed.`);
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(manifestBytes.toString("utf8"));
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
    !Array.isArray(parsed.shards) ||
    typeof parsed.contentSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(parsed.contentSha256)
  ) {
    throw new Error(`${label} packed ternary manifest is incompatible.`);
  }
  const expectedNames = parsed.coverage.eligibleTensorNames;
  if (
    expectedNames.some((name) => typeof name !== "string" || name.length === 0) ||
    new Set(expectedNames).size !== expectedNames.length ||
    parsed.coverage.eligibleTensorCount !== expectedNames.length ||
    parsed.tensors.length !== expectedNames.length
  ) {
    throw new Error(`${label} packed ternary coverage contract is invalid.`);
  }
  const shardFiles = new Set<string>();
  const shardPayloads = new Map<string, Buffer>();
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
      shardFiles.has(descriptor.file)
    ) {
      throw new Error(`${label} packed ternary shard table is invalid.`);
    }
    const payload = Buffer.from(files[descriptor.file] ?? []);
    if (
      payload.byteLength !== descriptor.byteLength ||
      sha256(payload) !== descriptor.sha256
    ) {
      throw new Error(`${label} packed ternary shard checksum failed.`);
    }
    shardFiles.add(descriptor.file);
    shardPayloads.set(descriptor.file, payload);
  }
  const tensorNames = new Set<string>();
  const referencedShards = new Set<string>();
  for (const tensor of parsed.tensors) {
    if (
      !isRecord(tensor) ||
      typeof tensor.name !== "string" ||
      typeof tensor.shard !== "string" ||
      !shardFiles.has(tensor.shard) ||
      tensorNames.has(tensor.name) ||
      referencedShards.has(tensor.shard) ||
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
    const payload = shardPayloads.get(tensor.shard);
    const expectedByteLength = Math.ceil(tensor.numel / 4);
    if (
      !Number.isSafeInteger(shapeProduct) ||
      shapeProduct !== tensor.numel ||
      !payload ||
      payload.byteLength !== expectedByteLength ||
      tensor.byteLength !== expectedByteLength ||
      tensor.packedSha256 !== sha256(payload)
    ) {
      throw new Error(`${label} packed ternary tensor shape or length is invalid.`);
    }
    const decoded = Buffer.alloc(tensor.numel);
    for (let index = 0; index < payload.byteLength * 4; index += 1) {
      const code = (payload[index >> 2]! >> ((index & 3) * 2)) & 0x03;
      if (index >= tensor.numel) {
        if (code !== 1) {
          throw new Error(`${label} packed ternary padding is non-canonical.`);
        }
        continue;
      }
      if (code === 3) {
        throw new Error(`${label} packed ternary data uses the reserved code.`);
      }
      decoded[index] = code === 0 ? 0xff : code === 1 ? 0 : 1;
    }
    const tensorDigest = sha256(
      Buffer.concat([
        Buffer.from(
          JSON.stringify({ dtype: "int8", shape: tensor.shape }),
          "utf8"
        ),
        Buffer.from([0]),
        decoded
      ])
    );
    if (tensorDigest !== tensor.tensorSha256) {
      throw new Error(`${label} packed ternary decoded tensor checksum failed.`);
    }
    tensorNames.add(tensor.name);
    referencedShards.add(tensor.shard);
  }
  if (
    tensorNames.size !== expectedNames.length ||
    expectedNames.some((name) => !tensorNames.has(name)) ||
    referencedShards.size !== shardFiles.size
  ) {
    throw new Error(`${label} packed ternary coverage is incomplete.`);
  }
  const allowedFiles = new Set(["manifest.json", "manifest.sha256", ...shardFiles]);
  if (Object.keys(files).some((name) => !allowedFiles.has(name))) {
    throw new Error(`${label} packed ternary directory contains an unlisted file.`);
  }
  return {
    manifestSha256: expectedManifestHash,
    tensorCount: expectedNames.length,
    files
  };
}

async function readPackedTernaryDirectory(
  directory: string,
  label: string
): Promise<PackedTernaryDirectory> {
  const directoryEntries = await readdir(directory, { withFileTypes: true });
  if (directoryEntries.some((entry) => !entry.isFile())) {
    throw new Error(`${label} packed ternary directory contains a non-file entry.`);
  }
  const files: Record<string, Uint8Array> = {};
  for (const entry of directoryEntries) {
    files[entry.name] = new Uint8Array(await readFile(join(directory, entry.name)));
  }
  return verifyPackedTernaryFiles(files, label);
}

function packedTernaryFilesFromBundle(
  files: Record<string, Uint8Array>,
  scope: "current" | "origin"
): Record<string, Uint8Array> {
  const prefix = `packed/${scope}/`;
  return Object.fromEntries(
    Object.entries(files)
      .filter(([path]) => path.startsWith(prefix))
      .map(([path, contents]) => [path.slice(prefix.length), contents])
  );
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
      try {
        await link(path, temporary);
      } catch (error) {
        const code = (error as NodeJS.ErrnoException).code;
        if (!["EXDEV", "EPERM", "EACCES", "ENOTSUP"].includes(code ?? "")) throw error;
        await copyFile(path, temporary);
      }
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
  ): Promise<PackedTernaryDirectory | undefined> {
    if (!(await pathExists(join(source, "manifest.json")))) return undefined;
    const packed = await readPackedTernaryDirectory(source, "Source");
    const temporary = `${destination}.${randomUUID()}.next`;
    const backup = `${destination}.${randomUUID()}.bak`;
    await mkdir(temporary, { recursive: true });
    try {
      for (const [name, contents] of Object.entries(packed.files)) {
        const hash = await this.storeBlob(Buffer.from(contents));
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
    await readPackedTernaryDirectory(destination, "Copied");
    return packed;
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
    await Promise.all([
      atomicWrite(join(targetEngine, "brain.json"), JSON.stringify(metadata, null, 2)),
      atomicWrite(join(targetOrigin, "brain.json"), JSON.stringify(metadata, null, 2))
    ]);
    const packedSource = join(sourceEngine, "packed-ternary");
    await Promise.all([
      this.copyPackedTernaryDirectory(
        packedSource,
        join(targetEngine, "packed-ternary")
      ),
      this.copyPackedTernaryDirectory(
        packedSource,
        join(targetOrigin, "packed-ternary")
      )
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
    await mkdir(join(directory, "snapshots"), { recursive: true });
    await atomicWrite(this.documentPath(fork.id), JSON.stringify(fork, null, 2));
    await writeFile(join(directory, "origin.json"), JSON.stringify(fork, null, 2), {
      encoding: "utf8",
      flag: "wx",
      mode: 0o600
    });
    await this.cloneEngineState(source.id, fork.id, fork.name);
    return clone(fork);
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
      await atomicWrite(join(engineSnapshot, "brain.json"), metadata);
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
      const hashes = [sha256(await readFile(join(engineSnapshot, "brain.json")))];
      for (const name of ["core.safetensors", "plasticity.safetensors"]) {
        const sourcePath = join(engineSnapshot, name);
        if (await pathExists(sourcePath)) hashes.push(sha256(await readFile(sourcePath)));
      }
      if (await pathExists(join(engineSnapshot, "packed-ternary", "manifest.json"))) {
        const packed = await readPackedTernaryDirectory(
          join(engineSnapshot, "packed-ternary"),
          "Snapshot"
        );
        hashes.push(packed.manifestSha256);
      }
      if (summary.engineChecksum && sha256(hashes.join(":")) !== summary.engineChecksum) {
        throw new Error("Neural snapshot checksum validation failed.");
      }
    }
    const saved = await this.save(restored);
    if (await pathExists(join(engineSnapshot, "brain.json"))) {
      const targetEngine = join(this.brainDirectory(id), "engine");
      const metadata = await readFile(join(engineSnapshot, "brain.json"));
      await atomicWrite(join(targetEngine, "brain.json"), metadata);
      for (const name of ["core.safetensors", "plasticity.safetensors"]) {
        const sourcePath = join(engineSnapshot, name);
        if (!(await pathExists(sourcePath))) continue;
        const hash = await this.storeFileAsBlob(sourcePath);
        await this.linkBlobTo(hash, join(targetEngine, name));
      }
      await this.copyPackedTernaryDirectory(
        join(engineSnapshot, "packed-ternary"),
        join(targetEngine, "packed-ternary")
      );
    }
    return saved;
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
        ? await readPackedTernaryDirectory(selectedPackedPath, "Current")
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
    const core = (await pathExists(corePath))
      ? new Uint8Array(await readFile(corePath))
      : validEmptySafetensors("Neural core has not been materialized by the Python worker.");
    const plasticity = (await pathExists(plasticityPath))
      ? new Uint8Array(await readFile(plasticityPath))
      : validEmptySafetensors("Plastic state is represented by state/brain.json.");
    assertSafeTensors(core, "core.safetensors");
    assertSafeTensors(plasticity, "plastic.safetensors");
    const entries: Record<string, Uint8Array> = {
      "model-card.md": strToU8(
        `# ${portableBrain.name}\n\nOmniCortex brain ${portableBrain.id}.\n\n` +
          `Preset: ${portableBrain.config.preset}\n\n` +
          `Memory recipe: ${portableBrain.config.memoryRecipe ?? "human-consolidation"}\n`
      ),
      "state/brain.json": strToU8(JSON.stringify(portableBrain, null, 2)),
      "state/engine.json": engineState,
      "tensors/core.safetensors": core,
      "tensors/plastic.safetensors": plasticity
    };
    if (selectedPacked) {
      for (const [name, contents] of Object.entries(selectedPacked.files)) {
        entries[`packed/current/${name}`] = contents;
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
    const immutableCore = (await pathExists(immutableCorePath))
      ? new Uint8Array(await readFile(immutableCorePath))
      : core;
    const immutablePlastic = (await pathExists(immutablePlasticPath))
      ? new Uint8Array(await readFile(immutablePlasticPath))
      : plasticity;
    const immutablePackedPath = join(immutableEngine, "packed-ternary");
    const immutablePacked =
      engineMaterialized &&
      (await pathExists(join(immutablePackedPath, "manifest.json")))
        ? await readPackedTernaryDirectory(immutablePackedPath, "Origin")
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
            currentCore: await this.storeBlob(Buffer.from(core)),
            currentPlasticity: await this.storeBlob(Buffer.from(plasticity)),
            originCore: await this.storeBlob(Buffer.from(immutableCore)),
            originPlasticity: await this.storeBlob(Buffer.from(immutablePlastic))
          }
        : undefined;
    if (references) {
      entries["tensors/core.safetensors"] = validEmptySafetensors(
        `Local content reference ${references.currentCore}`
      );
      entries["tensors/plastic.safetensors"] = validEmptySafetensors(
        `Local content reference ${references.currentPlasticity}`
      );
    }
    assertSafeTensors(immutableCore, "origin core.safetensors");
    assertSafeTensors(immutablePlastic, "origin plastic.safetensors");
    entries["origin/state/brain.json"] = strToU8(JSON.stringify(originBrain, null, 2));
    entries["origin/state/engine.json"] = immutableState;
    entries["origin/tensors/core.safetensors"] = references
      ? validEmptySafetensors(`Local content reference ${references.originCore}`)
      : immutableCore;
    entries["origin/tensors/plastic.safetensors"] = references
      ? validEmptySafetensors(`Local content reference ${references.originPlasticity}`)
      : immutablePlastic;
    if (immutablePacked) {
      for (const [name, contents] of Object.entries(immutablePacked.files)) {
        entries[`packed/origin/${name}`] = contents;
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
        for (const [name, contents] of Object.entries(packed.files)) {
          const hash = await this.storeBlob(Buffer.from(contents));
          packedReferences[scope][name] = hash;
          entries[`packed/${scope}/${name}`] = strToU8(
            `Local content reference ${hash}\n`
          );
        }
      }
    }
    for (const source of mode === "private-archive" ? portableBrain.trainingSources : []) {
      if (!source.blobHash || entries[`blobs/${source.blobHash}`]) continue;
      const blob = await this.getBlob(source.blobHash);
      const sample = blob.subarray(0, Math.min(blob.byteLength, 256 * 1024)).toString("utf8");
      const binaryRatio =
        sample.length === 0
          ? 0
          : ((sample.match(/\uFFFD/g)?.length ?? 0) + (sample.match(/\0/g)?.length ?? 0)) /
            sample.length;
      if (binaryRatio < 0.01) {
        const probe: RedactionCounter = { replacements: 0 };
        redactSecretText(blob.toString("utf8"), probe);
        if (probe.replacements > 0) {
          throw new Error(
            `${source.name} appears to contain credentials. Remove or sanitize it before a private archive export.`
          );
        }
      }
      entries[`blobs/${source.blobHash}`] = new Uint8Array(blob);
    }
    const fileRecords = Object.fromEntries(
      Object.entries(entries).map(([path, contents]) => [
        path,
        { sha256: sha256(Buffer.from(contents)), bytes: contents.byteLength }
      ])
    );
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
    entries["manifest.json"] = strToU8(JSON.stringify(manifest, null, 2));
    entries["checksums.sha256"] = strToU8(
      Object.entries(entries)
        .map(([path, contents]) => `${sha256(Buffer.from(contents))}  ${path}`)
        .sort()
        .join("\n") + "\n"
    );
    const archive = zipSync(entries, { level: 6 });
    await atomicWrite(destination, Buffer.from(archive));
  }

  async importBundle(path: string): Promise<BrainDocument> {
    const info = await stat(path);
    if (!info.isFile() || info.size > MAX_BUNDLE_BYTES) {
      throw new Error("The selected .omni bundle is not a supported size.");
    }
    return this.importBundleBuffer(await readFile(path), basename(path));
  }

  async importBundleBuffer(contents: Buffer, sourceLabel = "download.omni"): Promise<BrainDocument> {
    if (contents.byteLength > MAX_BUNDLE_BYTES) throw new Error("The .omni bundle is too large.");
    const directoryEntries = inspectZipDirectory(contents);
    let files: Record<string, Uint8Array>;
    try {
      files = unzipSync(new Uint8Array(contents));
    } catch {
      throw new Error(`${sourceLabel} is not a valid .omni ZIP container.`);
    }
    const names = new Set(directoryEntries.map((entry) => entry.name));
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
      if (!names.has(required) || !files[required]) {
        throw new Error(`The .omni bundle is missing ${required}.`);
      }
    }
    const checksums = parseChecksumFile(strFromU8(files["checksums.sha256"] as Uint8Array));
    for (const [path, expected] of checksums) {
      const file = files[path];
      if (!file) throw new Error(`Checksum references missing file ${path}.`);
      if (sha256(Buffer.from(file)) !== expected) {
        throw new Error(`Checksum validation failed for ${path}.`);
      }
    }
    for (const path of Object.keys(files)) {
      assertAllowedBundlePath(path);
      if (path !== "checksums.sha256" && !checksums.has(path)) {
        throw new Error(`The .omni bundle has no checksum for ${path}.`);
      }
    }
    let manifestValue: unknown;
    try {
      manifestValue = JSON.parse(strFromU8(files["manifest.json"] as Uint8Array));
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
      manifestValue.licenseLedger.sources.length > 100_000 ||
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
        typeof descriptor.bytes !== "number" ||
        !Number.isSafeInteger(descriptor.bytes) ||
        descriptor.bytes < 0
      ) {
        throw new Error(`Manifest descriptor for ${path} is invalid.`);
      }
      const file = files[path];
      if (
        !file ||
        file.byteLength !== descriptor.bytes ||
        sha256(Buffer.from(file)) !== descriptor.sha256
      ) {
        throw new Error(`Manifest checksum validation failed for ${path}.`);
      }
    }
    for (const path of Object.keys(files)) {
      if (path === "manifest.json" || path === "checksums.sha256") continue;
      if (!Object.hasOwn(manifestFiles, path)) {
        throw new Error(`Manifest is missing a descriptor for ${path}.`);
      }
    }
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
        try {
          files[path] = new Uint8Array(await this.getBlob(hash));
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
          if (!files[path]) {
            throw new Error(`The local referenced bundle is missing ${path}.`);
          }
          try {
            files[path] = new Uint8Array(await this.getBlob(hash));
          } catch {
            throw new Error(
              `Local packed ternary reference ${hash.slice(0, 12)}… is unavailable on this installation.`
            );
          }
        }
      }
    }
    let verifiedPackedCurrent: PackedTernaryDirectory | undefined;
    let verifiedPackedOrigin: PackedTernaryDirectory | undefined;
    const bundledCurrentPack = packedTernaryFilesFromBundle(files, "current");
    const bundledOriginPack = packedTernaryFilesFromBundle(files, "origin");
    if (packedDeclaration) {
      verifiedPackedCurrent = verifyPackedTernaryFiles(
        bundledCurrentPack,
        "Current bundle"
      );
      verifiedPackedOrigin = verifyPackedTernaryFiles(
        bundledOriginPack,
        "Origin bundle"
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
      Object.keys(bundledCurrentPack).length > 0 ||
      Object.keys(bundledOriginPack).length > 0
    ) {
      throw new Error("The .omni bundle contains undeclared packed ternary data.");
    }
    assertSafeTensors(files["tensors/core.safetensors"] as Uint8Array, "core.safetensors");
    assertSafeTensors(files["tensors/plastic.safetensors"] as Uint8Array, "plastic.safetensors");
    assertSafeTensors(
      files["origin/tensors/core.safetensors"] as Uint8Array,
      "origin core.safetensors"
    );
    assertSafeTensors(
      files["origin/tensors/plastic.safetensors"] as Uint8Array,
      "origin plastic.safetensors"
    );
    for (const [path, file] of Object.entries(files)) {
      if (!path.startsWith("blobs/")) continue;
      const hash = path.slice("blobs/".length);
      if (!/^[a-f0-9]{64}$/.test(hash) || sha256(Buffer.from(file)) !== hash) {
        throw new Error(`Content-addressed blob validation failed for ${path}.`);
      }
    }
    let brainValue: unknown;
    let originBrainValue: unknown;
    let engineValue: unknown;
    let originEngineValue: unknown;
    try {
      brainValue = JSON.parse(strFromU8(files["state/brain.json"] as Uint8Array));
      originBrainValue = JSON.parse(
        strFromU8(files["origin/state/brain.json"] as Uint8Array)
      );
      engineValue = JSON.parse(strFromU8(files["state/engine.json"] as Uint8Array));
      originEngineValue = JSON.parse(
        strFromU8(files["origin/state/engine.json"] as Uint8Array)
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
    await Promise.all([
      mkdir(join(directory, "snapshots"), { recursive: true }),
      mkdir(join(directory, "engine"), { recursive: true })
    ]);
    await atomicWrite(this.documentPath(imported.id), JSON.stringify(imported, null, 2));
    if (engineMaterialized) {
      if (resolvedReferences) {
        await Promise.all([
          atomicWrite(
            join(directory, "engine", "brain.json"),
            JSON.stringify(engineValue, null, 2)
          ),
          this.linkBlobTo(
            resolvedReferences.currentCore,
            join(directory, "engine", "core.safetensors")
          ),
          this.linkBlobTo(
            resolvedReferences.currentPlasticity,
            join(directory, "engine", "plasticity.safetensors")
          )
        ]);
      } else await Promise.all([
        atomicWrite(join(directory, "engine", "brain.json"), JSON.stringify(engineValue, null, 2)),
        atomicWrite(
          join(directory, "engine", "core.safetensors"),
          Buffer.from(files["tensors/core.safetensors"] as Uint8Array)
        ),
        atomicWrite(
          join(directory, "engine", "plasticity.safetensors"),
          Buffer.from(files["tensors/plastic.safetensors"] as Uint8Array)
        )
      ]);
    }
    if (engineMaterialized && verifiedPackedCurrent) {
      const packedDirectory = join(directory, "engine", "packed-ternary");
      await mkdir(packedDirectory, { recursive: true });
      for (const [name, contents] of Object.entries(verifiedPackedCurrent.files)) {
        const referenceHash = packedDeclaration?.references?.current[name];
        if (referenceHash) {
          await this.linkBlobTo(referenceHash, join(packedDirectory, name));
        } else {
          await atomicWrite(join(packedDirectory, name), Buffer.from(contents));
        }
      }
    }
    if (isRecord(originEngineValue) && originEngineValue.format === "omni-cortex-engine") {
      if (resolvedReferences) {
        await Promise.all([
          atomicWrite(
            join(directory, "engine", "origin", "brain.json"),
            JSON.stringify(originEngineValue, null, 2)
          ),
          this.linkBlobTo(
            resolvedReferences.originCore,
            join(directory, "engine", "origin", "core.safetensors")
          ),
          this.linkBlobTo(
            resolvedReferences.originPlasticity,
            join(directory, "engine", "origin", "plasticity.safetensors")
          )
        ]);
      } else await Promise.all([
        atomicWrite(
          join(directory, "engine", "origin", "brain.json"),
          JSON.stringify(originEngineValue, null, 2)
        ),
        atomicWrite(
          join(directory, "engine", "origin", "core.safetensors"),
          Buffer.from(files["origin/tensors/core.safetensors"] as Uint8Array)
        ),
        atomicWrite(
          join(directory, "engine", "origin", "plasticity.safetensors"),
          Buffer.from(files["origin/tensors/plastic.safetensors"] as Uint8Array)
        )
      ]);
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
      for (const [name, contents] of Object.entries(verifiedPackedOrigin.files)) {
        const referenceHash = packedDeclaration?.references?.origin[name];
        if (referenceHash) {
          await this.linkBlobTo(referenceHash, join(packedDirectory, name));
        } else {
          await atomicWrite(join(packedDirectory, name), Buffer.from(contents));
        }
      }
    }
    for (const [path, contents] of Object.entries(files)) {
      if (!path.startsWith("blobs/")) continue;
      await this.storeBlob(Buffer.from(contents));
    }
    await writeFile(join(directory, "origin.json"), JSON.stringify(importedOrigin, null, 2), {
      encoding: "utf8",
      flag: "wx",
      mode: 0o600
    });
    return clone(imported);
  }
}
