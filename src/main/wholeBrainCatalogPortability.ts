import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, realpath } from "node:fs/promises";
import { isAbsolute, join, relative as pathRelative, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import type { StreamingZipSource } from "./streamingZip";
import { sqliteFilename } from "./sqliteFilename";

const CATALOG_FORMAT = "omni-whole-brain-catalog";
const LINEAGE_FORMAT = "omni-whole-brain-catalog-lineage";
const LINEAGE_SHARD_FORMAT = "omni-whole-brain-catalog-lineage-shard";
const LINEAGE_SHARD_BYTES = 64 * 1024;
const SHA = /^[a-f0-9]{64}$/;
type Row = Record<string, unknown>;
export interface WholeBrainCatalogReadOptions {
  prefix?: string;
  signal?: AbortSignal;
  admit?(bytes: number, purpose: string): Promise<void> | void;
}
export async function admitWholeBrainCatalogMemory(options: WholeBrainCatalogReadOptions, bytes: number, purpose: string): Promise<void> {
  options.signal?.throwIfAborted();
  const inspect = (): void => {
    if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > freemem() || bytes > getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed)
      throw new Error(`Whole-brain catalog ${purpose} awaits physical/heap transfer memory.`);
  };
  inspect(); await options.admit?.(bytes, purpose); options.signal?.throwIfAborted(); inspect();
}
/** Exact-size guarded read before JSON decoding; a valid checksum is not RAM admission. */
export async function readWholeBrainCatalogMetadata(path: string, options: WholeBrainCatalogReadOptions = {},
  expected?: { bytes: number; sha256: string }): Promise<{ raw: Buffer; value: unknown }> {
  const before = await lstat(path, { bigint: true });
  const actualPath = await realpath(path);
  if (!before.isFile() || before.isSymbolicLink() || before.size > BigInt(Number.MAX_SAFE_INTEGER) ||
      expected && BigInt(expected.bytes) !== before.size) throw new Error("Unsafe or changed whole-brain JSON source.");
  const bytes = Number(before.size); await admitWholeBrainCatalogMemory(options, bytes * 8 + 65536, "JSON scratch");
  const input = await open(actualPath, constants.O_RDONLY | (process.platform === "win32" ? 0 : constants.O_NOFOLLOW));
  const identity = (value: typeof before) => [value.dev, value.ino, value.size, value.mtimeNs, value.ctimeNs].join(":");
  try {
    if (identity(await input.stat({ bigint: true })) !== identity(before)) throw new Error("Whole-brain JSON changed before its admitted read.");
    const raw = Buffer.alloc(bytes); let offset = 0;
    while (offset < bytes) { options.signal?.throwIfAborted(); const read = await input.read(raw, offset, Math.min(65536, bytes - offset), offset); if (!read.bytesRead) throw new Error("Whole-brain JSON ended before admitted bytes."); offset += read.bytesRead; }
    if (identity(await input.stat({ bigint: true })) !== identity(before) || identity(await lstat(path, { bigint: true })) !== identity(before) || await realpath(path) !== actualPath) throw new Error("Whole-brain JSON changed during its admitted read.");
    options.signal?.throwIfAborted(); if (expected && createHash("sha256").update(raw).digest("hex") !== expected.sha256) throw new Error("Whole-brain JSON checksum differs after admitted read.");
    return { raw, value: JSON.parse(raw.toString("utf8")) };
  } finally { await input.close(); }
}

export function safeWholeBrainCatalogPath(path: string): boolean {
  return /^whole-brain-catalog\/(?:blobs\/[a-f0-9]{64}\.json|index\/[a-f0-9]{64}\.sqlite3|lineage\/blobs\/[a-f0-9]{64}\.json|lineage\/sources\/[a-f0-9]{64}\/whole-brain-catalog\/(?:blobs\/[a-f0-9]{64}\.json|index\/[a-f0-9]{64}\.sqlite3))$/.test(path);
}

export interface WholeBrainCatalogFile {
  relativePath: string;
  sourcePath?: string;
  contents?: Uint8Array;
  bytes: number;
  sha256: string;
}
export interface WholeBrainCatalogClosure {
  sourceBrainId: string;
  sourceMetadataSha256: string;
  pointer: Row;
  files: Map<string, WholeBrainCatalogFile>;
}
export interface WholeBrainCatalogLineageEntry {
  entryKeySha256: string;
  entrySha256: string;
  sourceBrainId: string;
  sourceMetadataSha256: string;
  catalogBindingSha256: string;
  coreGeneration: string;
  substrateGeneration: string;
  closureRoot: string;
  catalogPointer: Row;
}
export interface WholeBrainCatalogLineageClosure {
  pointer?: Row;
  entries: Map<string, WholeBrainCatalogLineageEntry>;
  files: Map<string, WholeBrainCatalogFile>;
}
export interface WholeBrainCatalogAdoption {
  metadata: Row;
  files: Map<string, WholeBrainCatalogFile>;
  current: boolean;
}

const object = (value: unknown, label = "whole-brain catalog"): Row => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} is not an object.`);
  return value as Row;
};
const count = (value: unknown): value is number => typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
const text = (value: unknown, label: string): string => {
  if (typeof value !== "string" || !value || value.includes("\0")) throw new Error(`Invalid ${label}.`);
  return value;
};
const codepoint = (a: string, b: string): number => a < b ? -1 : a > b ? 1 : 0;
const exactKeys = (value: Row, names: readonly string[], label: string): void => {
  if (Object.keys(value).sort(codepoint).join("\0") !== [...names].sort(codepoint).join("\0"))
    throw new Error(`${label} shape differs.`);
};
export function wholeBrainCatalogCanonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(wholeBrainCatalogCanonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.entries(value as Row).sort(([a], [b]) => codepoint(a, b))
    .map(([key, item]) => `${JSON.stringify(key)}:${wholeBrainCatalogCanonical(item)}`).join(",")}}`;
  return JSON.stringify(value) ?? "null";
}
export function wholeBrainCatalogDigest(value: unknown): string {
  return createHash("sha256").update(wholeBrainCatalogCanonical(value)).digest("hex");
}
function safeRelative(value: unknown, pattern: RegExp): string {
  const path = text(value, "whole-brain catalog path");
  if (!pattern.test(path) || path.startsWith("/") || path.includes("\\") || path.split("/").some(part => !part || part === "." || part === ".."))
    throw new Error("Unsafe whole-brain catalog path.");
  return path;
}
async function safeFile(root: string, relative: string, expectedBytes: number | undefined,
  expectedSha256: string, signal?: AbortSignal): Promise<WholeBrainCatalogFile> {
  signal?.throwIfAborted();
  const base = await realpath(resolve(root)), path = join(base, ...relative.split("/"));
  const [before, actual] = await Promise.all([lstat(path), realpath(path)]);
  const owned = pathRelative(base, resolve(actual));
  if (!before.isFile() || before.isSymbolicLink() || resolve(actual) !== resolve(path) || !owned || owned.startsWith("..") || isAbsolute(owned) ||
      expectedBytes !== undefined && before.size !== expectedBytes)
    throw new Error("Whole-brain catalog source identity differs.");
  const handle = await open(path, constants.O_RDONLY | (process.platform === "win32" ? 0 : constants.O_NOFOLLOW));
  const identity = (value: Awaited<ReturnType<typeof lstat>>): string =>
    [value.dev, value.ino, value.size, value.mtimeMs, value.ctimeMs].join(":");
  try {
    const opened = await handle.stat();
    if (identity(opened) !== identity(before)) throw new Error("Whole-brain catalog changed while opening.");
    const digest = createHash("sha256");
    for await (const chunk of handle.createReadStream({ autoClose: false })) {
      signal?.throwIfAborted(); digest.update(chunk);
    }
    const [after, named, finalPath] = await Promise.all([handle.stat(), lstat(path), realpath(path)]);
    signal?.throwIfAborted();
    if (identity(after) !== identity(before) || identity(named) !== identity(before) || finalPath !== actual)
      throw new Error("Whole-brain catalog changed while reading.");
    const sha256 = digest.digest("hex");
    if (sha256 !== expectedSha256) throw new Error("Whole-brain catalog checksum differs.");
    return { relativePath: relative, sourcePath: path, bytes: before.size, sha256 };
  } finally { await handle.close(); }
}
function bindingFromMetadata(metadata: Row): Row {
  const brainId = text(metadata.brain_id, "whole-brain brain identity");
  const mutable = object(metadata.mutable_state ?? {}, "mutable-state pointer"), substrate = object(metadata.substrate ?? {}, "substrate pointer");
  const persistence = object(substrate.persistence ?? {}, "substrate persistence pointer");
  if (typeof mutable.activeGeneration !== "string" || !SHA.test(mutable.activeGeneration) ||
      typeof persistence.activeGeneration !== "string" || !SHA.test(persistence.activeGeneration))
    throw new Error("Whole-brain catalog needs committed core and substrate generations.");
  const descriptor = (value: unknown): string | null => {
    if (value === undefined || value === null) return null;
    const row = object(value, "connected descriptor"), checksum = row.sha256;
    if (typeof checksum !== "string" || !SHA.test(checksum)) throw new Error("Invalid connected descriptor checksum.");
    // Connected descriptors are sealed by the native source reader before a
    // catalog is published.  Do not re-encode the parsed value here: Python's
    // canonical JSON deliberately preserves numeric lexemes such as `1.0`,
    // while JavaScript would serialize the same parsed number as `1` and
    // falsely reject an authentic descriptor.  The catalog binding must still
    // name the exact authenticated descriptor checksum.
    return checksum;
  };
  return { brainId, coreGeneration: mutable.activeGeneration, substrateGeneration: persistence.activeGeneration,
    attentionOverlaySha256: wholeBrainCatalogDigest(metadata.attention_overlay ?? {}),
    connectedDescriptorSha256: descriptor(metadata.connected_regions),
    connectedGraphStatesSha256: descriptor(metadata.connected_graph_states) };
}
export function wholeBrainCatalogBinding(metadata: unknown): Record<string, unknown> { return bindingFromMetadata(object(metadata, "whole-brain binding metadata")); }
function checkedIndex(value: unknown): { path: string; sha256: string; bytes: number } {
  const row = object(value, "whole-brain index descriptor"), sha256 = row.sha256;
  exactKeys(row, ["path", "sha256", "bytes"], "Whole-brain catalog index descriptor");
  if (typeof sha256 !== "string" || !SHA.test(sha256) || row.path !== `whole-brain-catalog/index/${sha256}.sqlite3` ||
      !count(row.bytes) || row.bytes < 1) throw new Error("Whole-brain catalog index descriptor differs.");
  return { path: row.path as string, sha256, bytes: row.bytes };
}
function checkedPointer(value: unknown, binding?: Row): Row {
  const row = object(value, "whole-brain catalog pointer"), actualBinding = object(row.binding, "whole-brain binding");
  exactKeys(row, ["format", "version", "brainId", "binding", "bindingSha256", "tailSha256", "shards", "records",
    "counts", "index", "semanticAxisMapping", "activityObserved"], "Whole-brain catalog pointer");
  exactKeys(actualBinding, ["brainId", "coreGeneration", "substrateGeneration", "attentionOverlaySha256",
    "connectedDescriptorSha256", "connectedGraphStatesSha256"], "Whole-brain catalog binding");
  if (row.format !== CATALOG_FORMAT || row.version !== 1 || typeof row.brainId !== "string" ||
      binding && wholeBrainCatalogCanonical(actualBinding) !== wholeBrainCatalogCanonical(binding) ||
      actualBinding.brainId !== row.brainId || row.bindingSha256 !== wholeBrainCatalogDigest(actualBinding) ||
      typeof row.tailSha256 !== "string" || !SHA.test(row.tailSha256) || !count(row.shards) || row.shards < 1 ||
      !count(row.records) || row.semanticAxisMapping !== false || row.activityObserved !== false)
    throw new Error("Whole-brain catalog pointer identity differs.");
  const counts = object(row.counts, "whole-brain catalog counts"), names = ["regions", "owners", "aliases", "memberships", "bridges", "inspections"];
  if (Object.keys(counts).sort(codepoint).join("\0") !== [...names].sort(codepoint).join("\0") ||
      names.some(name => !count(counts[name])) || names.reduce((sum, name) => sum + Number(counts[name]), 0) !== row.records)
    throw new Error("Whole-brain catalog declared counts differ.");
  checkedIndex(row.index);
  return structuredClone(row);
}
export function wholeBrainCatalogSqlitePath(path: string, platform: NodeJS.Platform = process.platform): string {
  return sqliteFilename(path, platform);
}
async function validateIndex(file: WholeBrainCatalogFile, root: string, records: number, shards: Set<string>, options: WholeBrainCatalogReadOptions): Promise<void> {
  await admitWholeBrainCatalogMemory(options, 1024 * 1024, "SQLite/page scratch");
  const database = new DatabaseSync(sqliteFilename(file.sourcePath!), { readOnly: true, allowExtension: false });
  try {
    database.exec("PRAGMA trusted_schema=OFF; PRAGMA query_only=ON; PRAGMA cache_size=-512; PRAGMA mmap_size=0; PRAGMA temp_store=FILE; PRAGMA threads=1");
    if (database.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") throw new Error("Whole-brain catalog index integrity failed.");
    const columns = [...database.prepare("PRAGMA table_info(records)").iterate()].map(row => String(row.name));
    const expected = ["id", "kind", "shard", "ordinal", "owner_id", "region_id", "parent_id", "source_id", "target_id", "rank"];
    if (columns.join("\0") !== expected.join("\0") || Number(database.prepare("SELECT COUNT(*) AS value FROM records").get()?.value) !== records)
      throw new Error("Whole-brain catalog index schema/count differs.");
    let indexedShards = 0;
    for (const row of database.prepare("SELECT DISTINCT shard FROM records").iterate()) { options.signal?.throwIfAborted(); if (!shards.has(String(row.shard))) throw new Error("Whole-brain index references an incomplete shard set."); indexedShards++; }
    if (indexedShards !== shards.size) throw new Error("Whole-brain index references an incomplete shard set.");
    let active = "", actualRecords: unknown[] = [];
    for (const row of database.prepare("SELECT shard,ordinal,id FROM records ORDER BY shard,ordinal").iterate()) {
      options.signal?.throwIfAborted(); const shard = String(row.shard), ordinal = Number(row.ordinal), id = String(row.id);
      if (shard !== active) {
        active = shard;
        const relative = `whole-brain-catalog/blobs/${shard}.json`;
        const authenticated = await safeFile(root, relative, undefined, shard, options.signal);
        const body = object((await readWholeBrainCatalogMetadata(authenticated.sourcePath!, options, authenticated)).value);
        if (!Array.isArray(body.records)) throw new Error("Whole-brain indexed shard records are invalid.");
        actualRecords = body.records;
      }
      const value = actualRecords[ordinal];
      if (!recordId(value, id)) throw new Error("Whole-brain index location differs from its immutable record.");
    }
  } finally { database.close(); }
  await safeFile(root, file.relativePath, file.bytes, file.sha256, options.signal);
}
function recordId(value: unknown, id: string): boolean { return Boolean(value) && typeof value === "object" && !Array.isArray(value) && (value as Row).id === id; }
async function collectPointerFiles(root: string, pointerValue: unknown, options: WholeBrainCatalogReadOptions = {}): Promise<Map<string, WholeBrainCatalogFile>> {
  await admitWholeBrainCatalogMemory(options, 1024 * 1024, "pointer/index scratch");
  const pointer = checkedPointer(pointerValue), files = new Map<string, WholeBrainCatalogFile>();
  const index = checkedIndex(pointer.index);
  const indexFile = await safeFile(root, index.path, index.bytes, index.sha256, options.signal); files.set(index.path, indexFile);
  let cursor = pointer.tailSha256 as string | null, remaining = pointer.shards as number, rows = 0;
  const kinds: Readonly<Record<string, string>> = { region: "regions", owner: "owners", alias: "aliases",
    membership: "memberships", bridge: "bridges", inspection: "inspections" };
  const observedCounts: Record<string, number> = { regions: 0, owners: 0, aliases: 0, memberships: 0, bridges: 0, inspections: 0 };
  const shards = new Set<string>();
  while (cursor !== null) {
    options.signal?.throwIfAborted(); await admitWholeBrainCatalogMemory(options, 1024 * 1024 + (files.size + shards.size + 1) * 1024, "retained closure descriptors");
    if (!remaining || !SHA.test(cursor) || shards.has(cursor)) throw new Error("Whole-brain catalog chain is cyclic or exceeds its declaration.");
    shards.add(cursor);
    const relative = `whole-brain-catalog/blobs/${cursor}.json`, file = await safeFile(root, relative, undefined, cursor, options.signal);
    const body = object((await readWholeBrainCatalogMetadata(file.sourcePath!, options, file)).value, "whole-brain catalog shard");
    exactKeys(body, ["format", "version", "previousSha256", "records"], "Whole-brain catalog shard");
    if (body.format !== CATALOG_FORMAT || body.version !== 1 || !Array.isArray(body.records)) throw new Error("Whole-brain catalog shard format differs.");
    for (const record of body.records) {
      const row = object(record, "whole-brain catalog record"), name = typeof row.kind === "string" && Object.hasOwn(kinds, row.kind) ? kinds[row.kind] : undefined;
      if (!name || typeof row.id !== "string" || !row.id || row.id.includes("\0")) throw new Error("Whole-brain catalog record identity/kind differs.");
      observedCounts[name] = observedCounts[name]! + 1;
    }
    rows += body.records.length; files.set(relative, file);
    cursor = body.previousSha256 === null ? null : text(body.previousSha256, "whole-brain previous shard");
    remaining -= 1;
  }
  if (remaining || rows !== pointer.records) throw new Error("Whole-brain catalog closure lost records.");
  if (wholeBrainCatalogCanonical(observedCounts) !== wholeBrainCatalogCanonical(pointer.counts)) throw new Error("Whole-brain catalog record-kind counts differ from their declaration.");
  await validateIndex(indexFile, root, pointer.records as number, shards, options);
  return files;
}

export async function collectWholeBrainCatalog(engineRoot: string, metadataValue: unknown,
  options: WholeBrainCatalogReadOptions = {}): Promise<WholeBrainCatalogClosure | undefined> {
  const metadata = object(metadataValue, "whole-brain source metadata"), pointerValue = metadata.whole_brain_catalog;
  if (pointerValue === undefined) return undefined;
  const binding = bindingFromMetadata(metadata), pointer = checkedPointer(pointerValue, binding);
  const root = options.prefix ? join(engineRoot, ...options.prefix.split("/")) : engineRoot;
  return { sourceBrainId: binding.brainId as string, sourceMetadataSha256: wholeBrainCatalogDigest(metadata), pointer,
    files: await collectPointerFiles(root, pointer, options) };
}

function checkedLineagePointer(value: unknown): Row {
  const row = object(value, "whole-brain lineage pointer"), body = { format: LINEAGE_FORMAT, version: 1,
    tailSha256: row.tailSha256, shards: row.shards, entries: row.entries };
  exactKeys(row, ["format", "version", "tailSha256", "shards", "entries", "manifestSha256"], "Whole-brain lineage pointer");
  if (row.format !== LINEAGE_FORMAT || row.version !== 1 || typeof row.tailSha256 !== "string" || !SHA.test(row.tailSha256) ||
      !count(row.shards) || row.shards < 1 || !count(row.entries) || row.entries < 1 || row.manifestSha256 !== wholeBrainCatalogDigest(body))
    throw new Error("Whole-brain catalog lineage pointer differs.");
  return structuredClone(row);
}
function entryKey(value: Omit<WholeBrainCatalogLineageEntry, "entryKeySha256" | "entrySha256" | "closureRoot">): string {
  const index = checkedIndex(value.catalogPointer.index);
  return wholeBrainCatalogDigest({ sourceBrainId: value.sourceBrainId, sourceMetadataSha256: value.sourceMetadataSha256,
    catalogBindingSha256: value.catalogBindingSha256, tailSha256: value.catalogPointer.tailSha256, indexSha256: index.sha256 });
}
function checkedLineageEntry(value: unknown): WholeBrainCatalogLineageEntry {
  const row = object(value, "whole-brain lineage entry"), pointer = checkedPointer(row.catalogPointer);
  exactKeys(row, ["entryKeySha256", "entrySha256", "sourceBrainId", "sourceMetadataSha256", "catalogBindingSha256",
    "coreGeneration", "substrateGeneration", "closureRoot", "catalogPointer"], "Whole-brain lineage entry");
  for (const name of ["entryKeySha256", "entrySha256", "sourceMetadataSha256", "catalogBindingSha256", "coreGeneration", "substrateGeneration"])
    if (typeof row[name] !== "string" || !SHA.test(row[name] as string)) throw new Error("Whole-brain lineage digest is invalid.");
  const sourceBrainId = text(row.sourceBrainId, "whole-brain lineage source identity");
  if (pointer.brainId !== sourceBrainId || pointer.bindingSha256 !== row.catalogBindingSha256 ||
      (pointer.binding as Row).coreGeneration !== row.coreGeneration || (pointer.binding as Row).substrateGeneration !== row.substrateGeneration)
    throw new Error("Whole-brain lineage source binding differs.");
  const basis = { sourceBrainId, sourceMetadataSha256: row.sourceMetadataSha256 as string,
    catalogBindingSha256: row.catalogBindingSha256 as string, coreGeneration: row.coreGeneration as string,
    substrateGeneration: row.substrateGeneration as string, catalogPointer: pointer };
  const key = entryKey(basis), closureRoot = `whole-brain-catalog/lineage/sources/${key}`;
  const body = { entryKeySha256: key, sourceBrainId, sourceMetadataSha256: basis.sourceMetadataSha256,
    catalogBindingSha256: basis.catalogBindingSha256, coreGeneration: basis.coreGeneration,
    substrateGeneration: basis.substrateGeneration, closureRoot, catalogPointer: pointer };
  if (row.entryKeySha256 !== key || row.closureRoot !== closureRoot || row.entrySha256 !== wholeBrainCatalogDigest(body))
    throw new Error("Whole-brain lineage entry checksum differs.");
  return { ...body, entrySha256: row.entrySha256 as string };
}
function lineageEntry(current: WholeBrainCatalogClosure): WholeBrainCatalogLineageEntry {
  const binding = object(current.pointer.binding), basis = { sourceBrainId: current.sourceBrainId,
    sourceMetadataSha256: current.sourceMetadataSha256, catalogBindingSha256: current.pointer.bindingSha256 as string,
    coreGeneration: binding.coreGeneration as string, substrateGeneration: binding.substrateGeneration as string,
    catalogPointer: current.pointer };
  const key = entryKey(basis), closureRoot = `whole-brain-catalog/lineage/sources/${key}`;
  const body = { entryKeySha256: key, ...basis, closureRoot };
  return { ...body, entrySha256: wholeBrainCatalogDigest(body) };
}

export async function collectWholeBrainCatalogLineage(engineRoot: string, metadataValue: unknown,
  options: WholeBrainCatalogReadOptions = {}): Promise<WholeBrainCatalogLineageClosure> {
  const metadata = object(metadataValue), raw = metadata.whole_brain_catalog_lineage;
  if (raw === undefined) return { entries: new Map(), files: new Map() };
  const pointer = checkedLineagePointer(raw), root = options.prefix ? join(engineRoot, ...options.prefix.split("/")) : engineRoot;
  const files = new Map<string, WholeBrainCatalogFile>(), entries = new Map<string, WholeBrainCatalogLineageEntry>();
  let cursor = pointer.tailSha256 as string | null, remaining = pointer.shards as number, observed = 0;
  const seen = new Set<string>();
  while (cursor !== null) {
    if (!remaining || seen.has(cursor)) throw new Error("Whole-brain lineage chain is cyclic or exceeds its declaration.");
    seen.add(cursor); await admitWholeBrainCatalogMemory(options, 1024 * 1024 + (files.size + entries.size + seen.size) * 4096, "retained lineage descriptors"); const relative = `whole-brain-catalog/lineage/blobs/${cursor}.json`;
    const file = await safeFile(root, relative, undefined, cursor, options.signal);
    const body = object((await readWholeBrainCatalogMetadata(file.sourcePath!, options, file)).value);
    exactKeys(body, ["format", "version", "previousSha256", "entries"], "Whole-brain lineage shard");
    if (body.format !== LINEAGE_SHARD_FORMAT || body.version !== 1 || !Array.isArray(body.entries)) throw new Error("Whole-brain lineage shard format differs.");
    files.set(relative, file);
    for (const value of body.entries) {
      const entry = checkedLineageEntry(value), prior = entries.get(entry.entryKeySha256);
      if (prior && wholeBrainCatalogCanonical(prior) !== wholeBrainCatalogCanonical(entry)) throw new Error("Conflicting whole-brain lineage entry bytes.");
      entries.set(entry.entryKeySha256, entry); observed += prior ? 0 : 1;
    }
    cursor = body.previousSha256 === null ? null : text(body.previousSha256, "whole-brain lineage previous shard"); remaining -= 1;
  }
  if (remaining || observed !== pointer.entries) throw new Error("Whole-brain lineage closure count differs.");
  for (const entry of entries.values()) {
    const closure = await collectPointerFiles(join(root, ...entry.closureRoot.split("/")), entry.catalogPointer, options);
    for (const file of closure.values()) files.set(`${entry.closureRoot}/${file.relativePath}`, { ...file,
      relativePath: `${entry.closureRoot}/${file.relativePath}` });
  }
  return { pointer, entries, files };
}

async function buildLineage(entries: ReadonlyMap<string, WholeBrainCatalogLineageEntry>, options: WholeBrainCatalogReadOptions): Promise<{ pointer?: Row; files: Map<string, WholeBrainCatalogFile> }> {
  await admitWholeBrainCatalogMemory(options, 65536 + entries.size * 4096, "lineage publication descriptors");
  const ordered = [...entries.values()].sort((a, b) => codepoint(a.entryKeySha256, b.entryKeySha256));
  if (!ordered.length) return { files: new Map() };
  const files = new Map<string, WholeBrainCatalogFile>(); let previous: string | null = null, shards = 0, page: WholeBrainCatalogLineageEntry[] = [];
  const flush = (): void => {
    if (!page.length) return;
    const body = { format: LINEAGE_SHARD_FORMAT, version: 1, previousSha256: previous, entries: page }, encoded = Buffer.from(wholeBrainCatalogCanonical(body));
    const sha256 = createHash("sha256").update(encoded).digest("hex"), relativePath = `whole-brain-catalog/lineage/blobs/${sha256}.json`;
    files.set(relativePath, { relativePath, contents: encoded, bytes: encoded.byteLength, sha256 }); previous = sha256; shards += 1; page = [];
  };
  for (const entry of ordered) {
    const candidate = [...page, entry], size = Buffer.byteLength(wholeBrainCatalogCanonical({ format: LINEAGE_SHARD_FORMAT, version: 1, previousSha256: previous, entries: candidate }));
    if (page.length && size > LINEAGE_SHARD_BYTES) flush();
    page.push(entry);
  }
  flush();
  const body = { format: LINEAGE_FORMAT, version: 1, tailSha256: previous, shards, entries: ordered.length };
  return { pointer: { ...body, manifestSha256: wholeBrainCatalogDigest(body) }, files };
}

export async function adoptWholeBrainCatalog(sourceEngine: string, sourceMetadataValue: unknown,
  targetMetadataValue: unknown, targetBrainId: string,
  options: { sameIdentity: boolean; signal?: AbortSignal; sourcePrefix?: string; admit?: WholeBrainCatalogReadOptions["admit"] } = { sameIdentity: false }): Promise<WholeBrainCatalogAdoption> {
  const sourceMetadata = object(sourceMetadataValue), targetMetadata = structuredClone(object(targetMetadataValue));
  text(targetBrainId, "whole-brain target identity");
  const current = await collectWholeBrainCatalog(sourceEngine, sourceMetadata, { signal: options.signal, prefix: options.sourcePrefix, admit: options.admit });
  const inherited = await collectWholeBrainCatalogLineage(sourceEngine, sourceMetadata, { signal: options.signal, prefix: options.sourcePrefix, admit: options.admit });
  if (options.sameIdentity) {
    if (current && current.sourceBrainId !== targetBrainId) throw new Error("Same-identity catalog copy names another brain.");
    return { metadata: targetMetadata, files: new Map([...(current?.files ?? []), ...inherited.files]), current: Boolean(current) };
  }
  delete targetMetadata.whole_brain_catalog;
  const entries = new Map(inherited.entries);
  const files = new Map<string, WholeBrainCatalogFile>();
  for (const [relative, file] of inherited.files) if (!relative.startsWith("whole-brain-catalog/lineage/blobs/")) files.set(relative, file);
  if (current) {
    const entry = lineageEntry(current), prior = entries.get(entry.entryKeySha256);
    if (prior && wholeBrainCatalogCanonical(prior) !== wholeBrainCatalogCanonical(entry)) throw new Error("Conflicting adopted whole-brain source registry.");
    entries.set(entry.entryKeySha256, entry);
    for (const file of current.files.values()) {
      const relativePath = `${entry.closureRoot}/${file.relativePath}`;
      files.set(relativePath, { ...file, relativePath });
    }
  }
  const built = await buildLineage(entries, options);
  for (const [path, file] of built.files) files.set(path, file);
  if (built.pointer) targetMetadata.whole_brain_catalog_lineage = built.pointer;
  else delete targetMetadata.whole_brain_catalog_lineage;
  return { metadata: targetMetadata, files, current: false };
}

export function wholeBrainCatalogZipSources(closure: WholeBrainCatalogClosure | WholeBrainCatalogLineageClosure,
  prefix: string): StreamingZipSource[] {
  return [...closure.files.values()].sort((a, b) => codepoint(a.relativePath, b.relativePath)).map(file => ({
    name: `${prefix}/${file.relativePath}`, ...(file.sourcePath ? { sourcePath: file.sourcePath } : { contents: file.contents! })
  }));
}
